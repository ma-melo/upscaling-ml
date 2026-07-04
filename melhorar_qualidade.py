import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. COLE AQUI A ARQUITETURA DO SEU MODELO
# Para o PyTorch conseguir carregar os pesos (.pth), ele precisa saber 
# como é a "estrutura" do modelo. Cole as suas classes aqui:
# ResidualDenseBlock, RRDB, e ESRGANGenerator.
# =====================================================================
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


# =====================================================================
# 2. FUNÇÃO PRINCIPAL DE INFERÊNCIA
# =====================================================================
def aplicar_super_resolucao(caminho_imagem_entrada, caminho_imagem_saida, caminho_pesos, device):
    print(f"🔄 Carregando imagem: {caminho_imagem_entrada}")
    
    # 1. Abre a imagem original e garante que está em RGB
    img = Image.open(caminho_imagem_entrada).convert('RGB')
    
    # 2. Converte a imagem para Tensor (valores entre 0 e 1)
    transform_to_tensor = transforms.ToTensor()
    img_tensor = transform_to_tensor(img)
    
    # Adiciona a dimensão do "Batch" (lote). 
    # O PyTorch espera [Batch, Canais, Altura, Largura], então [3, H, W] vira [1, 3, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 3. Inicializa o modelo
    print("🧠 Carregando inteligência artificial...")
    model = ESRGANGenerator(scale=4).to(device)
    
    # Carrega os pesos salvos do seu treinamento (ex: esrgan_final_interpolated.pth)
    state_dict = torch.load(caminho_pesos, map_location=device)
    model.load_state_dict(state_dict)
    
    # Coloca o modelo em modo de avaliação (desliga cálculos de treino, otimizando memória)
    model.eval()
    
    # 4. Roda a imagem pelo modelo
    print("🚀 Aplicando super resolução 4x (isso pode levar alguns segundos)...")
    with torch.no_grad(): # Diz ao PyTorch para não calcular gradientes (economiza muita RAM/VRAM)
        output = model(img_tensor)
        
    # 5. Prepara a imagem de saída
    # Remove a dimensão do Batch, limita os pixels entre [0 e 1] e joga para CPU
    output_tensor = output.squeeze(0).clamp(0, 1).cpu()
    
    # Converte de Tensor de volta para Imagem PIL e salva no PC
    transform_to_image = transforms.ToPILImage()
    output_img = transform_to_image(output_tensor)
    
    output_img.save(caminho_imagem_saida)
    print(f"✅ Sucesso! Imagem salva em: {caminho_imagem_saida}")


# =====================================================================
# 3. CONFIGURAÇÕES PARA RODAR O SCRIPT
# =====================================================================
if __name__ == "__main__":
    # Verifica se tem Placa de Vídeo (CUDA) ou Mac (MPS), se não, usa CPU
    device = torch.device("cpu")
    
    # Mude estes caminhos para os arquivos do seu computador
    ENTRADA = "impa.png"
    SAIDA = "foto_antiga_4x.png" # Salvar como PNG evita perder qualidade na compressão
    PESOS = "resultados_esrgan/esrgan_final_interpolated.pth" # O arquivo gerado no seu notebook
    
    aplicar_super_resolucao(ENTRADA, SAIDA, PESOS, device)