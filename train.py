"""
Funções de treino e avaliação, reutilizáveis entre modelos (SRCNN, EDSR, ESRGAN etc.)

Este módulo tem 3 funções, cada uma com uma responsabilidade bem definida:
    train_one_epoch -> roda uma época de treino (forward + backward + optimizer.step)
    evaluate        -> avalia o modelo em um loader (PSNR/SSIM médios, sem gradiente)
    fit             -> orquestra o loop completo de treino, chamando as duas acima
                       a cada época e cuidando de scheduler, TensorBoard, checkpoint
                       do melhor modelo e early stopping.
"""

import statistics
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import calc_psnr, calc_ssim


def _upscale_to_match(lr_img, hr_img):
    """Upsamplea lr_img via bicubic para o tamanho espacial exato de hr_img.

    Usamos `size=hr_img.shape[-2:]` em vez de `scale_factor=scale` porque
    scale_factor multiplica o tamanho ATUAL do tensor -- se o LR não tiver sido
    gerado por uma divisão inteira exata (ex: patch_size=33, scale=4, onde
    33 // 4 = 8 e 8 * 4 = 32 != 33), o resultado não bate com o tamanho do HR
    e a loss quebra por incompatibilidade de shape. Mirar no tamanho absoluto
    do HR é robusto a qualquer combinação de patch_size/scale.
    """
    return F.interpolate(lr_img, size=hr_img.shape[-2:], mode="bicubic", align_corners=False)


def train_one_epoch(
    model, loader, optimizer, criterion, device,
    scale=4, upscale_input=True, desc="Treinando",
    grad_clip=None, use_amp=False, scaler=None,
):
    """Treina o modelo por uma época.

    Parâmetros
    ----------
    upscale_input : bool
        True  -> usado pela SRCNN (recebe o LR já upscalado via bicubic).
        False -> usado por modelos que fazem upsampling internamente (EDSR, ESRGAN).
    desc : str
        Rótulo mostrado na barra de progresso (tqdm). Útil para diferenciar qual
        modelo/escala está treinando quando há múltiplos treinos em sequência
        (ex: "Treinando [edsr_scale4]").
    grad_clip : float | None
        Se definido, aplica clipagem de norma de gradiente
        (torch.nn.utils.clip_grad_norm_) com esse valor máximo. Recomendado para
        redes mais profundas (EDSR, ESRGAN), onde picos de gradiente podem
        desestabilizar o treino. None desativa (comportamento original).
    use_amp : bool
        Se True, usa mixed precision (torch.cuda.amp) para acelerar o treino em
        GPU CUDA (sem efeito relevante em CPU/MPS). Requer `scaler`.
    scaler : torch.cuda.amp.GradScaler | None
        Obrigatório se use_amp=True. Deve ser criado uma única vez fora do loop
        de épocas (mantém estado entre chamadas) -- fit() já cuida disso.

    Retorna
    -------
    float : loss média da época (média das médias por batch; se o último batch
    for menor que os demais, ele pesa igual aos outros -- viés pequeno, mas
    existente caso len(dataset) não seja múltiplo de batch_size).
    """
    if use_amp and scaler is None:
        raise ValueError("use_amp=True requer um `scaler` (torch.cuda.amp.GradScaler) criado fora do loop.")

    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for lr_img, hr_img in progress_bar:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)

        inp = _upscale_to_match(lr_img, hr_img) if upscale_input else lr_img

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                sr = model(inp)
                loss = criterion(sr, hr_img)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            sr = model(inp)
            loss = criterion(sr, hr_img)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")

    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, scale=4, upscale_input=True, ssim_fn=calc_ssim,
             criterion=None, show_progress=True):
    """Avalia o modelo em um dataset de teste (ex: Set5, Set14, DIV2K_valid).

    IMPORTANTE: assume loader com batch_size=1. Cada métrica é calculada por
    imagem individual e depois agregada. Com batch_size > 1, PSNR/SSIM seriam
    calculados sobre o batch inteiro de uma vez (média entre as imagens do
    batch), perdendo granularidade por imagem -- por isso o uso recomendado
    é sempre batch_size=1 neste loader.

    Parâmetros
    ----------
    ssim_fn : callable
        Função usada para calcular o SSIM. Por padrão, `calc_ssim` (versão
        global simplificada, rápida -- adequada para acompanhar métricas
        durante o treino, mas NÃO é o SSIM padrão da literatura). Para os
        resultados finais do relatório (comparáveis com os papers), passe
        `calc_ssim_skimage`:
            from metrics import calc_ssim_skimage
            evaluate(model, loader, device, ssim_fn=calc_ssim_skimage)
    criterion : callable | None
        Se fornecido (a mesma loss usada no treino, ex: nn.L1Loss()), calcula
        também a "loss de validação" -- no mesmo formato/escala da train_loss,
        permitindo comparar as duas curvas e identificar overfitting (loss de
        treino caindo enquanto a de validação estagna ou sobe). Calculada
        sobre a saída do modelo ANTES do clamp(0,1), pra ficar no mesmo
        espaço numérico da loss vista durante o treino. Se None (padrão), a
        chave "loss_val" não aparece no retorno.
    show_progress : bool
        Mostra uma barra de progresso (tqdm). Diferente do treino, aqui cada
        "batch" é uma imagem inteira (não um patch pequeno), então avaliar um
        benchmark maior (ex: Urban100) pode demorar visivelmente sem feedback.
        fit() chama evaluate() com show_progress=False para não competir com
        a barra de progresso do treino.

    Retorna
    -------
    dict com psnr_model, psnr_model_std, ssim_model, ssim_model_std, psnr_bicubic,
    e opcionalmente loss_val (se `criterion` for passado). Os campos *_std
    (desvio padrão populacional) ajudam a identificar se o modelo é consistente
    entre as imagens do benchmark ou se poucas imagens "difíceis" estão
    puxando a média para baixo -- útil para a discussão dos resultados.
    """
    model.eval()

    psnr_model, ssim_model = [], []
    psnr_bicubic = []
    loss_val = [] if criterion is not None else None

    iterator = tqdm(loader, desc="Avaliando", leave=False) if show_progress else loader

    for lr_img, hr_img in iterator:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)

        lr_up = _upscale_to_match(lr_img, hr_img)
        inp = lr_up if upscale_input else lr_img

        sr_raw = model(inp)

        if criterion is not None:
            loss_val.append(criterion(sr_raw, hr_img).item())

        sr = sr_raw.clamp(0, 1)

        psnr_model.append(calc_psnr(sr, hr_img))
        ssim_model.append(ssim_fn(sr, hr_img))
        psnr_bicubic.append(calc_psnr(lr_up, hr_img))

    result = {
        "psnr_model": sum(psnr_model) / len(psnr_model),
        "psnr_model_std": statistics.pstdev(psnr_model),
        "ssim_model": sum(ssim_model) / len(ssim_model),
        "ssim_model_std": statistics.pstdev(ssim_model),
        "psnr_bicubic": sum(psnr_bicubic) / len(psnr_bicubic),
    }
    if loss_val is not None:
        result["loss_val"] = sum(loss_val) / len(loss_val)

    return result


def fit(
    model, train_loader, val_loader, optimizer, criterion, device, epochs,
    scale=4, upscale_input=True, scheduler=None, scheduler_needs_metric=False,
    eval_every=5, writer=None, tag="", save_best_path=None,
    grad_clip=None, use_amp=False, patience=None, train_desc=None,
):
    """Loop de treino completo para super-resolução.

    Parâmetros
    ----------
    eval_every : int
        Frequência (em épocas) de chamadas a `evaluate`. A avaliação roda em
        imagens inteiras (não patches), então é mais cara que uma época de
        treino -- por padrão, avalia a cada 5 épocas. A última época é sempre
        avaliada, independente de eval_every, para garantir uma métrica final.
        Use eval_every=1 para reproduzir o comportamento "avaliar toda época".
    scheduler_needs_metric : bool
        Schedulers como ReduceLROnPlateau exigem uma métrica em
        `scheduler.step(metrica)`, diferente de schedulers "cegos" como StepLR
        (que usam `scheduler.step()` sem argumento). Se True, fit() chama
        `scheduler.step(val_metrics["psnr_model"])` -- e só nas épocas em que
        avaliação de fato ocorreu (eval_every); nas demais épocas, o
        scheduler simplesmente não avança (não há métrica nova pra basear a
        decisão).
    writer : SummaryWriter | None
        Se fornecido, loga loss/PSNR/SSIM/learning rate no TensorBoard a cada
        época, usando `tag` para diferenciar múltiplos treinos no mesmo writer.
    tag : str
        Sufixo usado nas chaves do TensorBoard e na barra de progresso
        (ex: "srcnn", "edsr_scale4").
    save_best_path : str | Path | None
        Se fornecido, salva o state_dict do modelo sempre que o PSNR de
        validação melhora em relação ao melhor até então -- evita que o
        checkpoint salvo seja o da última época caso o modelo já tenha
        começado a piorar (overfitting) antes do fim do treino.
    grad_clip, use_amp : repassados diretamente para train_one_epoch.
    patience : int | None
        Se fornecido, interrompe o treino (early stopping) caso o PSNR de
        validação não melhore por `patience` avaliações consecutivas.
        None (padrão) desativa -- treina sempre até `epochs`.
    train_desc : str | None
        Rótulo da barra de progresso do treino. Por padrão, usa
        f"Treinando [{tag}]" se `tag` for fornecida, senão "Treinando".

    Retorna
    -------
    history : dict com:
        'train_loss'    -> lista com 1 valor por época (sempre).
        'epoch_time_sec'-> duração de cada época em segundos (treino + avaliação,
            quando houve avaliação naquela época) -- 1 valor por época.
        'psnr_model', 'ssim_model', 'psnr_bicubic', 'val_loss' -> 1 valor por
            AVALIAÇÃO (não por época, caso eval_every > 1). 'val_loss' só é
            preenchida se um `criterion` para a validação estiver disponível
            (fit() reaproveita o mesmo `criterion` do treino).
        'eval_epochs'   -> em quais números de época cada avaliação ocorreu,
            para alinhar os campos acima ao eixo x correto num gráfico
            (ex: plt.plot(history["eval_epochs"], history["val_loss"])).
    """
    history = {
        "train_loss": [],
        "epoch_time_sec": [],
        "psnr_model": [],
        "ssim_model": [],
        "psnr_bicubic": [],
        "val_loss": [],
        "eval_epochs": [],
    }

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    desc = train_desc if train_desc is not None else (f"Treinando [{tag}]" if tag else "Treinando")

    best_psnr = -float("inf")
    epochs_sem_melhora = 0

    def _key(base):
        return f"{base}_{tag}" if tag else base

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scale=scale, upscale_input=upscale_input, desc=desc,
            grad_clip=grad_clip, use_amp=use_amp, scaler=scaler,
        )
        history["train_loss"].append(train_loss)

        is_last_epoch = (epoch == epochs - 1)
        should_eval = ((epoch + 1) % eval_every == 0) or is_last_epoch

        val_metrics = None
        if should_eval:
            val_metrics = evaluate(
                model, val_loader, device,
                scale=scale, upscale_input=upscale_input,
                criterion=criterion, show_progress=False,
            )
            history["psnr_model"].append(val_metrics["psnr_model"])
            history["ssim_model"].append(val_metrics["ssim_model"])
            history["psnr_bicubic"].append(val_metrics["psnr_bicubic"])
            history["val_loss"].append(val_metrics["loss_val"])
            history["eval_epochs"].append(epoch + 1)

        epoch_time = time.time() - epoch_start
        history["epoch_time_sec"].append(epoch_time)

        # --- scheduler ---
        if scheduler is not None:
            if scheduler_needs_metric:
                if val_metrics is not None:
                    scheduler.step(val_metrics["psnr_model"])
                # se não avaliou nesta época, não há métrica nova -> scheduler não avança
            else:
                scheduler.step()

        # --- melhor checkpoint + early stopping ---
        if val_metrics is not None:
            melhorou = val_metrics["psnr_model"] > best_psnr
            if melhorou:
                best_psnr = val_metrics["psnr_model"]
                epochs_sem_melhora = 0
                if save_best_path is not None:
                    torch.save(model.state_dict(), save_best_path)
            else:
                epochs_sem_melhora += 1

        # --- TensorBoard ---
        if writer is not None:
            writer.add_scalar(_key("Loss/train"), train_loss, epoch)
            writer.add_scalar(_key("Time/epoch_sec"), epoch_time, epoch)
            if val_metrics is not None:
                writer.add_scalar(_key("PSNR/val"), val_metrics["psnr_model"], epoch)
                writer.add_scalar(_key("SSIM/val"), val_metrics["ssim_model"], epoch)
                writer.add_scalar(_key("PSNR/bicubic"), val_metrics["psnr_bicubic"], epoch)
                writer.add_scalar(_key("Loss/val"), val_metrics["loss_val"], epoch)

        # --- log no console ---
        if val_metrics is not None:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss:.6f} (val: {val_metrics['loss_val']:.6f}) | "
                f"PSNR: {val_metrics['psnr_model']:.2f} dB "
                f"(bicubic: {val_metrics['psnr_bicubic']:.2f} dB) | "
                f"SSIM: {val_metrics['ssim_model']:.4f} | "
                f"Tempo: {epoch_time:.1f}s"
            )
        else:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {train_loss:.6f} | Tempo: {epoch_time:.1f}s")

        # --- early stopping ---
        if patience is not None and val_metrics is not None and epochs_sem_melhora >= patience:
            print(f"Early stopping: sem melhora de PSNR por {patience} avaliações consecutivas.")
            break

    return history