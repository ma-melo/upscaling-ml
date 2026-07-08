# funcoes de treino e avaliacao, reutilizaveis entre modelos (srcnn, edsr, esrgan etc)
#
# train_one_epoch -> roda uma epoca de treino (forward + backward + optimizer.step)
# evaluate -> avalia o modelo em um loader (psnr/ssim medios, sem gradiente)
# fit -> orquestra o loop completo, chamando as duas acima a cada epoca e cuidando de scheduler, tensorboard, checkpoint do melhor modelo e early stopping

import statistics
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import calc_psnr, calc_ssim


def _upscale_to_match(lr_img, hr_img):
    # upsamplea lr_img via bicubic pro tamanho espacial exato de hr_img
    #
    # usa size=hr_img.shape[-2:] em vez de scale_factor porque scale_factor multiplica o tamanho atual do tensor
    # se o lr nao veio de uma divisao inteira exata (ex: patch_size=33, scale=4, 33 // 4 = 8, 8 * 4 = 32 != 33)
    # o resultado nao bate com o hr e a loss quebra por shape incompativel
    # mirar no tamanho absoluto do hr e robusto pra qualquer combinacao
    return F.interpolate(lr_img, size=hr_img.shape[-2:], mode="bicubic", align_corners=False)


def train_one_epoch(
    model, loader, optimizer, criterion, device,
    scale=4, upscale_input=True, desc="treinando",
    grad_clip=None, use_amp=False, scaler=None,
):
    # treina o modelo por uma epoca
    #
    # upscale_input: true pra srcnn (recebe lr ja upscalado via bicubic), false pra modelos que fazem upsampling interno (edsr, esrgan)
    # grad_clip: se definido, aplica clip_grad_norm_ com esse valor maximo.
    #   recomendado pra redes mais profundas (edsr, esrgan), onde picos de gradiente podem desestabilizar o treino
    # use_amp: mixed precision (torch.cuda.amp), so faz diferenca em gpu cuda.
    #   requer scaler, criado uma unica vez fora do loop de epocas (fit ja cuida disso)
    #
    # retorna a loss media da epoca (media das medias por batch)
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
    # avalia o modelo num dataset de teste (ex: set5, set14, div2k_valid)
    #
    # assume loader com batch_size=1: cada metrica e calculada por imagem individual e depois agregada
    # com batch_size > 1 o psnr/ssim seria calculado sobre o batch inteiro, perdendo granularidade por imagem
    #
    # ssim_fn: por padrao calc_ssim (versao global simplificada, rapida,
    #   boa pra acompanhar o treino, mas nao e o ssim padrao da literatura)
    #   pros resultados finais do relatorio, passar calc_ssim_skimage:
    #       from metrics import calc_ssim_skimage
    #       evaluate(model, loader, device, ssim_fn=calc_ssim_skimage)
    # criterion: se fornecido, calcula tambem a loss de validacao no mesmo formato da train_loss, pra comparar as duas curvas e identificar overfitting.
    #   calculada antes do clamp(0,1), no mesmo espaco numerico da loss vista no treino. se none, a chave "loss_val" nao aparece
    # show_progress: mostra barra de progresso. aqui cada "batch" e uma imagem inteira, entao um benchmark maior (urban100) pode demorar sem feedback. fit() chama com show_progress=False
    #
    # retorna dict com psnr_model, psnr_model_std, ssim_model, ssim_model_std, psnr_bicubic e opcionalmente loss_val
    # os campos *_std ajudam a ver se o modelo e consistente entre as imagens ou se poucas imagens dificeis puxam a media pra baixo
    model.eval()

    psnr_model, ssim_model = [], []
    psnr_bicubic = []
    loss_val = [] if criterion is not None else None

    iterator = tqdm(loader, desc="avaliando", leave=False) if show_progress else loader

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
    # loop de treino completo pra super-resolucao
    #
    # eval_every: frequencia (em epocas) de chamadas a evaluate. a avaliacao
    #   roda em imagens inteiras, entao e mais cara que uma epoca de treino
    #   -- por padrao avalia a cada 5 epocas. a ultima epoca sempre e
    #   avaliada, independente de eval_every
    # scheduler_needs_metric: reducelronplateau exige scheduler.step(metrica),
    #   diferente de schedulers "cegos" como steplr (scheduler.step() sem
    #   argumento). se true, so avanca nas epocas em que houve avaliacao
    # writer: se fornecido, loga loss/psnr/ssim/lr no tensorboard a cada
    #   epoca, usando `tag` pra diferenciar treinos no mesmo writer
    # save_best_path: se fornecido, salva o state_dict sempre que o psnr de
    #   validacao melhora -- evita salvar o checkpoint da ultima epoca caso
    #   o modelo ja tenha comecado a piorar (overfitting)
    # patience: se fornecido, para o treino (early stopping) caso o psnr de
    #   validacao nao melhore por `patience` avaliacoes consecutivas
    # train_desc: rotulo da barra de progresso. padrao usa "treinando [tag]"
    #   se tag for fornecida, senao "treinando"
    #
    # retorna history com:
    #   train_loss      -> 1 valor por epoca
    #   epoch_time_sec  -> duracao de cada epoca em segundos, 1 por epoca
    #   psnr_model, ssim_model, psnr_bicubic, val_loss -> 1 valor por
    #     avaliacao (nao por epoca, se eval_every > 1)
    #   eval_epochs     -> em quais epocas cada avaliacao ocorreu, pra
    #     alinhar os campos acima no eixo x de um grafico
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
    desc = train_desc if train_desc is not None else (f"treinando [{tag}]" if tag else "treinando")

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

        # scheduler
        if scheduler is not None:
            if scheduler_needs_metric:
                if val_metrics is not None:
                    scheduler.step(val_metrics["psnr_model"])
                # se nao avaliou nesta epoca, scheduler nao avanca
            else:
                scheduler.step()

        # melhor checkpoint + early stopping
        if val_metrics is not None:
            melhorou = val_metrics["psnr_model"] > best_psnr
            if melhorou:
                best_psnr = val_metrics["psnr_model"]
                epochs_sem_melhora = 0
                if save_best_path is not None:
                    torch.save(model.state_dict(), save_best_path)
            else:
                epochs_sem_melhora += 1

        # tensorboard
        if writer is not None:
            writer.add_scalar(_key("Loss/train"), train_loss, epoch)
            writer.add_scalar(_key("Time/epoch_sec"), epoch_time, epoch)
            if val_metrics is not None:
                writer.add_scalar(_key("PSNR/val"), val_metrics["psnr_model"], epoch)
                writer.add_scalar(_key("SSIM/val"), val_metrics["ssim_model"], epoch)
                writer.add_scalar(_key("PSNR/bicubic"), val_metrics["psnr_bicubic"], epoch)
                writer.add_scalar(_key("Loss/val"), val_metrics["loss_val"], epoch)

        # log no console
        if val_metrics is not None:
            print(
                f"epoch {epoch + 1}/{epochs} | "
                f"loss: {train_loss:.6f} (val: {val_metrics['loss_val']:.6f}) | "
                f"psnr: {val_metrics['psnr_model']:.2f} db "
                f"(bicubic: {val_metrics['psnr_bicubic']:.2f} db) | "
                f"ssim: {val_metrics['ssim_model']:.4f} | "
                f"tempo: {epoch_time:.1f}s"
            )
        else:
            print(f"epoch {epoch + 1}/{epochs} | loss: {train_loss:.6f} | tempo: {epoch_time:.1f}s")

        # early stopping
        if patience is not None and val_metrics is not None and epochs_sem_melhora >= patience:
            print(f"early stopping: sem melhora de psnr por {patience} avaliacoes consecutivas.")
            break

    return history