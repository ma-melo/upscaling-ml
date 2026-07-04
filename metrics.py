"""
Métricas de avaliação para super-resolução.

MÉTRICAS PADRÃO (RGB):
- calc_psnr()           -> PSNR em RGB
- calc_ssim()           -> SSIM simplificado em RGB (rápido, treino)
- calc_ssim_skimage()   -> SSIM janelado via skimage em RGB (mais preciso)

MÉTRICAS DO PAPERS (Canal Y - Luminância):
- calc_psnr_y()         -> PSNR no canal Y (padrão dos papers)
- calc_ssim_y()         -> SSIM janelado no canal Y (padrão dos papers)

MÉTRICAS COM CROP DE BORDA:
- calc_psnr_y_border()  -> PSNR Y ignorando borda (evita penalizar artefatos)
- calc_ssim_y_border()  -> SSIM Y ignorando borda

MÉTRICA PERCEPTUAL:
- calc_lpips()          -> LPIPS (melhor correlação com percepção humana)
"""

import torch
import torch.nn.functional as F


def _rgb_to_y(rgb):
    """Converte tensor RGB para canal Y (luminância) em formato YCbCr.

    Fórmula padrão ITU-R BT.601:
    Y = 0.299*R + 0.587*G + 0.114*B

    Parâmetros
    ----------
    rgb : torch.Tensor
        Tensor RGB em [0, 1], shape [B, 3, H, W] ou [3, H, W]

    Retorna
    -------
    torch.Tensor
        Canal Y, shape [B, 1, H, W] ou [1, H, W]
    """
    # Adiciona batch dimension se necessário
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)

    # Conversão ITU-R BT.601
    y = 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]
    return y


def _crop_border(img, border_size):
    """Remove borda de uma imagem.

    Prática comum em SR para evitar penalizar artefatos de borda.

    Parâmetros
    ----------
    img : torch.Tensor
        Tensor [B, C, H, W]
    border_size : int
        Tamanho da borda a remover em cada lado

    Retorna
    -------
    torch.Tensor
        Imagem com borda removida
    """
    if border_size == 0:
        return img
    b = border_size
    return img[:, :, b:-b, b:-b]


# ============================================================================
# MÉTRICAS BÁSICAS (RGB)
# ============================================================================

def calc_psnr(sr, hr, max_val=1.0):
    """PSNR em RGB. Rápido, usado durante treino."""
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10((max_val ** 2) / mse).item()


def calc_ssim(sr, hr, C1=0.01 ** 2, C2=0.03 ** 2):
    """SSIM simplificado, global (não janelado) em RGB.

    Útil para monitorar o treino rapidamente, mas NÃO é o SSIM padrão da
    literatura (que é calculado em janelas locais). Para resultados
    finais do relatório, prefira calc_ssim_y() ou calc_ssim_skimage().
    """
    mu_sr, mu_hr = sr.mean(), hr.mean()
    var_sr, var_hr = sr.var(), hr.var()
    cov = ((sr - mu_sr) * (hr - mu_hr)).mean()

    ssim = ((2 * mu_sr * mu_hr + C1) * (2 * cov + C2)) / (
        (mu_sr ** 2 + mu_hr ** 2 + C1) * (var_sr + var_hr + C2)
    )
    return ssim.item()


def calc_ssim_skimage(sr, hr):
    """SSIM janelado via skimage em RGB.

    sr, hr: tensores [1, C, H, W] ou [C, H, W] em [0, 1].
    Mais preciso que calc_ssim(), mas mais lento.
    """
    from skimage.metrics import structural_similarity as ssim_fn

    # Garante formato correto
    if sr.dim() == 4:
        sr = sr.squeeze(0)
    if hr.dim() == 4:
        hr = hr.squeeze(0)

    sr_np = sr.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    hr_np = hr.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    return ssim_fn(sr_np, hr_np, channel_axis=2, data_range=1.0)


# ============================================================================
# MÉTRICAS NO CANAL Y (PADRÃO DOS PAPERS)
# ============================================================================

def calc_psnr_y(sr, hr, max_val=1.0):
    """PSNR no canal Y (luminância) - PADRÃO DOS PAPERS.

    Métrica padrão usada em papers de SR (Dong, Lim, Wang).
    Calcula PSNR apenas no canal Y, ignorando Cb/Cr.

    Parâmetros
    ----------
    sr, hr : torch.Tensor
        Tensores RGB em [0, 1], shape [B, 3, H, W]
    max_val : float
        Valor máximo (1.0 para [0, 1], 255.0 para [0, 255])

    Retorna
    -------
    float
        PSNR em dB
    """
    sr_y = _rgb_to_y(sr)
    hr_y = _rgb_to_y(hr)

    mse = F.mse_loss(sr_y, hr_y)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10((max_val ** 2) / mse).item()


def calc_ssim_y(sr, hr):
    """SSIM janelado no canal Y (luminância) - PADRÃO DOS PAPERS.

    Métrica padrão usada em papers de SR.
    Converte para Y, depois calcula SSIM via skimage (mais preciso).

    Parâmetros
    ----------
    sr, hr : torch.Tensor
        Tensores RGB em [0, 1], shape [B, 3, H, W] ou [3, H, W]

    Retorna
    -------
    float
        SSIM no intervalo [-1, 1]
    """
    from skimage.metrics import structural_similarity as ssim_fn

    sr_y = _rgb_to_y(sr).squeeze(1)  # [B, H, W]
    hr_y = _rgb_to_y(hr).squeeze(1)  # [B, H, W]

    # Calcula SSIM na primeira imagem do batch (ou única imagem)
    sr_np = sr_y[0].cpu().numpy().clip(0, 1)
    hr_np = hr_y[0].cpu().numpy().clip(0, 1)

    return ssim_fn(sr_np, hr_np, data_range=1.0)


# ============================================================================
# MÉTRICAS COM CROP DE BORDA
# ============================================================================

def calc_psnr_y_border(sr, hr, border_size=4, max_val=1.0):
    """PSNR no canal Y ignorando borda.

    Prática comum em SR: remove pixels de borda (tamanho = scale factor)
    para evitar penalizar artefatos inerentes de borda.

    Parâmetros
    ----------
    sr, hr : torch.Tensor
        Tensores RGB em [0, 1]
    border_size : int
        Pixels a remover de cada lado (padrão: 4 para scale=4x)
    max_val : float
        Valor máximo

    Retorna
    -------
    float
        PSNR em dB
    """
    sr_y = _rgb_to_y(sr)
    hr_y = _rgb_to_y(hr)

    sr_y = _crop_border(sr_y, border_size)
    hr_y = _crop_border(hr_y, border_size)

    mse = F.mse_loss(sr_y, hr_y)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10((max_val ** 2) / mse).item()


def calc_ssim_y_border(sr, hr, border_size=4):
    """SSIM no canal Y ignorando borda.

    Parâmetros
    ----------
    sr, hr : torch.Tensor
        Tensores RGB em [0, 1]
    border_size : int
        Pixels a remover de cada lado

    Retorna
    -------
    float
        SSIM
    """
    from skimage.metrics import structural_similarity as ssim_fn

    sr_y = _rgb_to_y(sr).squeeze(1)  # [B, H, W]
    hr_y = _rgb_to_y(hr).squeeze(1)  # [B, H, W]

    sr_y = _crop_border(sr_y.unsqueeze(1), border_size).squeeze(1)
    hr_y = _crop_border(hr_y.unsqueeze(1), border_size).squeeze(1)

    sr_np = sr_y[0].cpu().numpy().clip(0, 1)
    hr_np = hr_y[0].cpu().numpy().clip(0, 1)

    return ssim_fn(sr_np, hr_np, data_range=1.0)


# ============================================================================
# MÉTRICAS PERCEPTUAIS (Opcional - requer torchvision)
# ============================================================================

def calc_lpips(sr, hr, net='alex', version='0.1'):
    """LPIPS (Learned Perceptual Image Patch Similarity).

    Métrica perceptual que correlaciona MUITO melhor com percepção humana
    que PSNR/SSIM. Usa features pré-treinadas de uma rede neural.

    Requer: pip install lpips

    Parâmetros
    ----------
    sr, hr : torch.Tensor
        Tensores RGB em [0, 1], shape [B, 3, H, W]
    net : str
        Rede backbone: 'alex', 'vgg', 'squeeze'
    version : str
        Versão do modelo

    Retorna
    -------
    float
        LPIPS score (0-1, lower is better)
    """
    try:
        import lpips
        device = sr.device
        loss_fn = lpips.LPIPS(net=net, version=version).to(device)
        with torch.no_grad():
            score = loss_fn(sr, hr).item()
        return score
    except ImportError:
        print("⚠️  LPIPS não instalado. Use: pip install lpips")
        return None


if __name__ == "__main__":
    """Teste rápido das métricas."""
    print("Testando métricas...")

    # Dummy data
    sr = torch.rand(1, 3, 64, 64)
    hr = torch.rand(1, 3, 64, 64)

    print(f"PSNR (RGB):           {calc_psnr(sr, hr):.2f} dB")
    print(f"PSNR (Y):             {calc_psnr_y(sr, hr):.2f} dB")
    print(f"PSNR (Y + border):    {calc_psnr_y_border(sr, hr, border_size=4):.2f} dB")
    print(f"SSIM (RGB):           {calc_ssim(sr, hr):.4f}")
    print(f"SSIM (Y):             {calc_ssim_y(sr, hr):.4f}")
    print(f"SSIM (Y + border):    {calc_ssim_y_border(sr, hr, border_size=4):.4f}")
    print("✅ Todas as métricas funcionando!")