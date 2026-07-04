import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path

#Caminhos
IMAGEM = "datasets/DIV2K_valid_HR/0804.png"
CAMINHO_PESOS = "resultados_esrgan/esrgan_final_interpolated.pth"
PASTA_SAIDA = Path("imagens_saida")
PASTA_SAIDA.mkdir(exist_ok=True)

#=========================
# ESRGAN
#=========================

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        return (self.RDB3(self.RDB2(self.RDB1(x)))) * 0.2 + x

class ESRGANGenerator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.HRconv(fea)))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=========================
# Saídas
#=========================

def rodar_super_resolucao():
    if not os.path.exists(CAMINHO_PESOS):
        print(f"Arquivo de pesos não encontrado em '{CAMINHO_PESOS}'.")
        return

    model = ESRGANGenerator(scale=4).to(device)
    model.load_state_dict(torch.load(CAMINHO_PESOS, map_location=device))
    model.eval()
    
    if not os.path.exists(IMAGEM):
        print(f"Imagem não encontrada em '{IMAGEM}'.")
        return
        
    #Imagem original
    img_hr_pil = Image.open(IMAGEM).convert("RGB")
    w, h = img_hr_pil.size
    
    #Imagem LR
    img_lr_pil = img_hr_pil.resize((w // 4, h // 4), Image.BICUBIC)
    
    lr_tensor = TF.to_tensor(img_lr_pil).unsqueeze(0).to(device)
    
    #ESRGAN
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1) 
    
    #Bicubic
    lr_upscaled_tensor = F.interpolate(lr_tensor, size=(h, w), mode="bicubic", align_corners=False)
    
    #Salvando as 4 imagens
    nome_arquivo = Path(IMAGEM).stem
    
    save_image(TF.to_tensor(img_hr_pil), PASTA_SAIDA / f"{nome_arquivo}_original_HR.png")
    save_image(TF.to_tensor(img_lr_pil), PASTA_SAIDA / f"{nome_arquivo}_baixa_resolucao_LR.png")
    save_image(lr_upscaled_tensor.squeeze(0), PASTA_SAIDA / f"{nome_arquivo}_upscale_bicubic.png")
    save_image(sr_tensor.squeeze(0), PASTA_SAIDA / f"{nome_arquivo}_resultado_esrgan.png")
    
    print(f"\nImagens salvas na pasta '{PASTA_SAIDA}':")
    print(f"Original (HR):      {nome_arquivo}_original_HR.png")
    print(f"Baixa Resolução:    {nome_arquivo}_baixa_resolucao_LR.png")
    print(f"Ampliada Bicubic:   {nome_arquivo}_upscale_bicubic.png")
    print(f"Ampliada ESRGAN:    {nome_arquivo}_resultado_esrgan.png")

if __name__ == "__main__":
    rodar_super_resolucao()