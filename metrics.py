# metricas de avaliacao para super-resolucao
#
# metricas rgb: calc_psnr, calc_ssim, calc_ssim_skimage
# metricas canal y (padrao dos papers): calc_psnr_y, calc_ssim_y
# metricas com crop de borda: calc_psnr_y_border, calc_ssim_y_border
# metrica perceptual: calc_lpips

import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size=11, sigma=1.5, device=None, dtype=None):
    # kernel gaussiano 2d normalizado, usado no ssim janelado
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
    return g2d.unsqueeze(0).unsqueeze(0)


def _ssim_windowed(img1, img2, data_range=1.0, window_size=11, sigma=1.5):
    # ssim janelado via convolucao gaussiana, sem depender do skimage
    # janela 11x11, sigma 1.5, sem padding -> equivalente ao skimage
    # com gaussian_weights=True, sigma=1.5, use_sample_covariance=False
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    C = img1.shape[1]
    window = _gaussian_kernel(window_size, sigma, device=img1.device, dtype=img1.dtype)
    window = window.expand(C, 1, window_size, window_size)

    # sem padding (valid), igual ao comportamento padrao do skimage
    mu1 = F.conv2d(img1, window, groups=C)
    mu2 = F.conv2d(img2, window, groups=C)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=C) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def _rgb_to_y(rgb):
    # converte rgb para canal y (luminancia), formato ycbcr
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)

    # conversao itu-r bt.601
    y = 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]
    return y


def _crop_border(img, border_size):
    # remove borda da imagem, pratica comum em sr
    if border_size == 0:
        return img
    b = border_size
    return img[:, :, b:-b, b:-b]

# metricas basicas (rgb)

def calc_psnr(sr, hr, max_val=1.0):
    # psnr em rgb, usado durante treino
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10((max_val ** 2) / mse).item()


def calc_ssim(sr, hr, C1=0.01 ** 2, C2=0.03 ** 2):
    # ssim simplificado, global, em rgb
    # rapido pro treino, mas nao e o ssim janelado da literatura
    # pra relatorio final, usar calc_ssim_y ou calc_ssim_skimage
    mu_sr, mu_hr = sr.mean(), hr.mean()
    var_sr, var_hr = sr.var(), hr.var()
    cov = ((sr - mu_sr) * (hr - mu_hr)).mean()

    ssim = ((2 * mu_sr * mu_hr + C1) * (2 * cov + C2)) / (
        (mu_sr ** 2 + mu_hr ** 2 + C1) * (var_sr + var_hr + C2)
    )
    return ssim.item()


def calc_ssim_skimage(sr, hr):
    # ssim janelado em rgb, mais preciso e mais lento que calc_ssim
    sr = sr.clamp(0, 1)
    hr = hr.clamp(0, 1)
    return _ssim_windowed(sr, hr, data_range=1.0)


# metricas no canal y (padrao dos papers)

def calc_psnr_y(sr, hr, max_val=1.0):
    # psnr no canal y, padrao usado em papers de sr (dong, lim, wang)
    sr_y = _rgb_to_y(sr)
    hr_y = _rgb_to_y(hr)

    mse = F.mse_loss(sr_y, hr_y)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10((max_val ** 2) / mse).item()


def calc_ssim_y(sr, hr):
    # ssim janelado no canal y, padrao usado em papers de sr
    sr_y = _rgb_to_y(sr)
    hr_y = _rgb_to_y(hr)

    # usa a primeira imagem do batch
    sr_y0 = sr_y[0:1].clamp(0, 1)
    hr_y0 = hr_y[0:1].clamp(0, 1)

    return _ssim_windowed(sr_y0, hr_y0, data_range=1.0)


# metricas com crop de borda

def calc_psnr_y_border(sr, hr, border_size=4, max_val=1.0):
    # psnr no canal y ignorando borda (tamanho = fator de escala)
    sr_y = _rgb_to_y(sr)
    hr_y = _rgb_to_y(hr)

    sr_y = _crop_border(sr_y, border_size)
    hr_y = _crop_border(hr_y, border_size)

    mse = F.mse_loss(sr_y, hr_y)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10((max_val ** 2) / mse).item()


def calc_ssim_y_border(sr, hr, border_size=4):
    # ssim no canal y ignorando borda
    sr_y = _rgb_to_y(sr)
    hr_y = _rgb_to_y(hr)

    sr_y = _crop_border(sr_y, border_size)
    hr_y = _crop_border(hr_y, border_size)

    sr_y0 = sr_y[0:1].clamp(0, 1)
    hr_y0 = hr_y[0:1].clamp(0, 1)

    return _ssim_windowed(sr_y0, hr_y0, data_range=1.0)


# metrica perceptual (opcional, requer lpips)

def calc_lpips(sr, hr, net='alex', version='0.1'):
    # lpips, correlaciona melhor com percepcao humana que psnr/ssim
    # requer: pip install lpips
    try:
        import lpips
        device = sr.device
        loss_fn = lpips.LPIPS(net=net, version=version).to(device)
        with torch.no_grad():
            score = loss_fn(sr, hr).item()
        return score
    except ImportError:
        print("lpips nao instalado. use: pip install lpips")
        return None


if __name__ == "__main__":
    # teste rapido das metricas
    sr = torch.rand(1, 3, 64, 64)
    hr = torch.rand(1, 3, 64, 64)

    print(f"psnr (rgb): {calc_psnr(sr, hr):.2f} db")
    print(f"psnr (y): {calc_psnr_y(sr, hr):.2f} db")
    print(f"psnr (y + border): {calc_psnr_y_border(sr, hr, border_size=4):.2f} db")
    print(f"ssim (rgb): {calc_ssim(sr, hr):.4f}")
    print(f"ssim (y): {calc_ssim_y(sr, hr):.4f}")
    print(f"ssim (y + border): {calc_ssim_y_border(sr, hr, border_size=4):.4f}")