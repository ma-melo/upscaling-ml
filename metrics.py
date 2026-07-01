"""
Métricas de avaliação para super-resolução.

calc_psnr      -> PSNR padrão.
calc_ssim      -> versão global simplificada (rápida, útil para acompanhar treino).
calc_ssim_skimage -> versão janelada via skimage, usar para os resultados finais
                     do relatório (mais correta e comparável com os papers).
"""

import torch
import torch.nn.functional as F


def calc_psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10((max_val ** 2) / mse).item()


def calc_ssim(sr, hr, C1=0.01 ** 2, C2=0.03 ** 2):
    """SSIM simplificado, global (não janelado).

    Útil para monitorar o treino rapidamente, mas NÃO é o SSIM padrão da
    literatura (que é calculado em janelas locais). Para os resultados
    finais do relatório, prefira calc_ssim_skimage.
    """
    mu_sr, mu_hr = sr.mean(), hr.mean()
    var_sr, var_hr = sr.var(), hr.var()
    cov = ((sr - mu_sr) * (hr - mu_hr)).mean()

    ssim = ((2 * mu_sr * mu_hr + C1) * (2 * cov + C2)) / (
        (mu_sr ** 2 + mu_hr ** 2 + C1) * (var_sr + var_hr + C2)
    )
    return ssim.item()


def calc_ssim_skimage(sr, hr):
    """SSIM janelado via skimage. sr, hr: tensores [1, C, H, W] em [0, 1].

    Requer: pip install scikit-image --break-system-packages
    """
    from skimage.metrics import structural_similarity as ssim_fn

    sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    hr_np = hr.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    return ssim_fn(sr_np, hr_np, channel_axis=2, data_range=1.0)