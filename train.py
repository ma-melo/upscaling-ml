"""
Funções de treino e avaliação, reutilizáveis entre modelos (SRCNN, EDSR, etc.)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import calc_psnr, calc_ssim


def train_one_epoch(model, loader, optimizer, criterion, device, scale=4, upscale_input=True):
    """Treina o modelo por uma época.

    upscale_input=True  -> usado pela SRCNN (recebe o LR já upscalado via bicubic)
    upscale_input=False -> usado por modelos que fazem o upsampling internamente (ex: EDSR)
    """
    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(loader, desc="Treinando", leave=False)

    for lr_img, hr_img in progress_bar:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)

        if upscale_input:
            inp = F.interpolate(lr_img, scale_factor=scale, mode="bicubic", align_corners=False)
        else:
            inp = lr_img

        optimizer.zero_grad()
        sr = model(inp)
        loss = criterion(sr, hr_img)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, scale=4, upscale_input=True):
    """Avalia o modelo em um dataset de teste (ex: Set5, Set14).

    Retorna um dicionário com PSNR/SSIM médios do modelo e do bicubic puro
    (baseline "sem rede"), para facilitar a comparação.
    """
    model.eval()

    psnr_model, ssim_model = [], []
    psnr_bicubic = []

    for lr_img, hr_img in loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)

        lr_up = F.interpolate(lr_img, scale_factor=scale, mode="bicubic", align_corners=False)
        inp = lr_up if upscale_input else lr_img

        sr = model(inp).clamp(0, 1)

        psnr_model.append(calc_psnr(sr, hr_img))
        ssim_model.append(calc_ssim(sr, hr_img))
        psnr_bicubic.append(calc_psnr(lr_up, hr_img))

    return {
        "psnr_model": sum(psnr_model) / len(psnr_model),
        "ssim_model": sum(ssim_model) / len(ssim_model),
        "psnr_bicubic": sum(psnr_bicubic) / len(psnr_bicubic),
    }


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, scale=4, upscale_input=True, scheduler=None,):
    """Loop de treino completo para super-resolução.

    Parâmetros
    ----------
    scale, upscale_input : repassados para train_one_epoch e evaluate.
    scheduler            : chamado ao final de cada época (opcional).

    Retorna
    -------
    history : dict com listas 'train_loss', 'psnr_model', 'ssim_model', 'psnr_bicubic'.
    """
    history = {
        "train_loss": [],
        "psnr_model": [],
        "ssim_model": [],
        "psnr_bicubic": [],
    }

    for epoch in range(epochs):
        # train_one_epoch retorna apenas o loss médio da época
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scale=scale, upscale_input=upscale_input,
        )

        # evaluate retorna dict {psnr_model, ssim_model, psnr_bicubic}
        val_metrics = evaluate(
            model, val_loader, device,
            scale=scale, upscale_input=upscale_input,
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["psnr_model"].append(val_metrics["psnr_model"])
        history["ssim_model"].append(val_metrics["ssim_model"])
        history["psnr_bicubic"].append(val_metrics["psnr_bicubic"])

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.6f} | "
            f"PSNR: {val_metrics['psnr_model']:.2f} dB "
            f"(bicubic: {val_metrics['psnr_bicubic']:.2f} dB) | "
            f"SSIM: {val_metrics['ssim_model']:.4f}"
        )

    return history