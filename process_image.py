import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path

from models import SRCNN, EDSRBaseline, ESRGANGenerator

# caminhos
IMAGEM = "imagens_teste/impa.png" # aqui vc altera para a imagem que deseja testar
PESOS_SRCNN = "outputs/SRCNN_2026-07-07/srcnn_final.pt"
PESOS_EDSR = "outputs/EDSR_2026-07-07/edsr_final.pt"
PESOS_ESRGAN = "outputs/ESRGAN_2026-07-07/esrgan_final_interpolated.pth"
PESOS_ESRGAN_DF2K = "outputs/ESRGAN_DF2K_2026-07-07/esrgan_final_interpolated.pth"
PASTA_SAIDA = Path("imagens_saida")
PASTA_SAIDA.mkdir(exist_ok=True)

device = torch.device("cpu")

#=========================
# Saídas
#=========================

def rodar_super_resolucao():
    if not os.path.exists(IMAGEM):
        print(f"Imagem não encontrada em '{IMAGEM}'.")
        return

    # imagem original
    img_hr_pil = Image.open(IMAGEM).convert("RGB")
    w, h = img_hr_pil.size

    # imagem LR
    img_lr_pil = img_hr_pil.resize((w // 4, h // 4), Image.BICUBIC)

    lr_tensor = TF.to_tensor(img_lr_pil).unsqueeze(0).to(device)

    # Bicubic
    lr_upscaled_tensor = F.interpolate(lr_tensor, size=(h, w), mode="bicubic", align_corners=False)

    # SRCNN (precisa de LR já upscalado via bicubic)
    srcnn_tensor = None
    if os.path.exists(PESOS_SRCNN):
        srcnn_model = SRCNN().to(device)
        srcnn_model.load_state_dict(torch.load(PESOS_SRCNN, map_location=device))
        srcnn_model.eval()
        with torch.no_grad():
            srcnn_tensor = srcnn_model(lr_upscaled_tensor)
            srcnn_tensor = torch.clamp(srcnn_tensor, 0, 1)

    # EDSR (faz upsampling internamente)
    edsr_tensor = None
    if os.path.exists(PESOS_EDSR):
        edsr_model = EDSRBaseline(num_features=64, num_blocks=16, scale=4, res_scale=0.1).to(device)
        edsr_model.load_state_dict(torch.load(PESOS_EDSR, map_location=device))
        edsr_model.eval()
        with torch.no_grad():
            edsr_tensor = edsr_model(lr_tensor)
            edsr_tensor = torch.clamp(edsr_tensor, 0, 1)

    # ESRGAN
    esrgan_tensor = None
    if os.path.exists(PESOS_ESRGAN):
        esrgan_model = ESRGANGenerator(scale=4).to(device)
        esrgan_model.load_state_dict(torch.load(PESOS_ESRGAN, map_location=device))
        esrgan_model.eval()
        with torch.no_grad():
            esrgan_tensor = esrgan_model(lr_tensor)
            esrgan_tensor = torch.clamp(esrgan_tensor, 0, 1)

    # ESRGAN treinado com o DF2K
    esrgan_tensor_df2k = None
    if os.path.exists(PESOS_ESRGAN_DF2K):
        esrgan_model_df2k = ESRGANGenerator(scale=4).to(device)
        esrgan_model_df2k.load_state_dict(torch.load(PESOS_ESRGAN_DF2K, map_location=device))
        esrgan_model_df2k.eval()
        with torch.no_grad():
            esrgan_tensor_df2k = esrgan_model_df2k(lr_tensor)
            esrgan_tensor_df2k = torch.clamp(esrgan_tensor_df2k, 0, 1)

    # salvando as imagens
    nome_arquivo = Path(IMAGEM).stem

    save_image(TF.to_tensor(img_hr_pil), PASTA_SAIDA / f"{nome_arquivo}_original_HR.png")
    save_image(TF.to_tensor(img_lr_pil), PASTA_SAIDA / f"{nome_arquivo}_baixa_resolucao_LR.png")
    save_image(lr_upscaled_tensor.squeeze(0), PASTA_SAIDA / f"{nome_arquivo}_upscale_bicubic.png")

    if srcnn_tensor is not None:
        save_image(srcnn_tensor.squeeze(0), PASTA_SAIDA / f"{nome_arquivo}_resultado_srcnn.png")

    if edsr_tensor is not None:
        save_image(edsr_tensor.squeeze(0), PASTA_SAIDA / f"{nome_arquivo}_resultado_edsr.png")

    if esrgan_tensor is not None:
        save_image(esrgan_tensor.squeeze(0), PASTA_SAIDA / f"{nome_arquivo}_resultado_esrgan.png")

    if esrgan_tensor_df2k is not None:
        save_image(esrgan_tensor_df2k.squeeze(0), PASTA_SAIDA / f"{nome_arquivo}_resultado_esrgan_df2k.png")

    print(f"\nImagens salvas na pasta '{PASTA_SAIDA}':")
    print(f"Original (HR): {nome_arquivo}_original_HR.png")
    print(f"Baixa Resolução: {nome_arquivo}_baixa_resolucao_LR.png")
    print(f"Ampliada Bicubic: {nome_arquivo}_upscale_bicubic.png")
    if srcnn_tensor is not None:
        print(f"Ampliada SRCNN: {nome_arquivo}_resultado_srcnn.png")
    if edsr_tensor is not None:
        print(f"Ampliada EDSR: {nome_arquivo}_resultado_edsr.png")
    if esrgan_tensor is not None:
        print(f"Ampliada ESRGAN: {nome_arquivo}_resultado_esrgan.png")
    if esrgan_tensor_df2k is not None:
        print(f"Ampliada ESRGAN DF2K: {nome_arquivo}_resultado_esrgan_df2k.png")

if __name__ == "__main__":
    rodar_super_resolucao()