# Super-Resolução de Imagens: Comparação de Arquiteturas

**Projeto Final da Disciplina [ML2](https://ganacim.github.io/ml2-2026/)** (2026)  
**Alunos:** [Marcos Abílio](https://github.com/ma-melo) e [Gabriel Vieira](https://github.com/gabrielgszv)

---

## 📋 Introdução

Este trabalho implementa, treina e compara três arquiteturas clássicas de super-resolução (SR) de imagens: **SRCNN**, **EDSR-baseline** e **ESRGAN**. O objetivo é avaliar o trade-off entre métricas tradicionais (PSNR, SSIM) e qualidade perceptual, respondendo a pergunta fundamental: *quanto de ganho métrico reflete ganho visual real?*

---

## 🎯 O Problema: Super-Resolução de Imagens

A super-resolução é o problema de aumentar a resolução de uma imagem de baixa resolução (LR) para alta resolução (HR), reconstruindo detalhes que foram perdidos durante a degradação. É um problema **mal-posto** (ill-posed) porque múltiplas imagens HR podem produzir a mesma imagem LR.

### Aplicações práticas
- Melhoramento de câmeras de vigilância
- Aumento de resolução de imagens históricas/médicas
- Preparação de conteúdo para exibição em telas maiores

### Desafio técnico
O espaço de possíveis soluções é enorme. Modelos ingênuos (interpolação bicúbica) produzem imagens borradas. Redes neurais aprendem a alucinarem detalhes realistas, mas com trade-offs:
- **Fidelidade pixel-a-pixel** (PSNR alto) vs. **Qualidade perceptual** (menos artefatos, texturas mais naturais)

---

## 📚 Arquiteturas de Referência

### 1. **SRCNN** (Dong et al., 2016)

**Referência:** [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.04112)

- **Conceito:** Primeira rede deep learning para SR; simples e elegante
- **Arquitetura:** 3 camadas convolucionais (9→1→5 kernels)
- **Entrada:** Imagem LR já upscalada para tamanho HR via bicúbica
- **Vantagem:** Rápida, baixo uso de memória
- **Limitação:** Sem residuais; dificuldade em treinar profundidade

```
Input (bicúbic upscaled) → Conv(9) → ReLU → Conv(1) → ReLU → Conv(5) → Output
```

### 2. **EDSR-baseline** (Lim et al., 2017)

**Referência:** [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02671)

- **Conceito:** Arquitetura residual moderna; treina em RGB end-to-end
- **Arquitetura:** 16 blocos residuais (64 features), sem BatchNorm, upsampling via PixelShuffle
- **Entrada:** Imagem LR "pequena"; a rede aprende upsampling internamente
- **Vantagem:** Melhor performance (PSNR), treino mais estável, sem BatchNorm (menos dependência de batch size)
- **Inovação chave:** Skip connection global ao redor dos blocos residuais

```
Input (LR) → Conv(3) → [16 ResidualBlocks] → Conv(3) + Skip → PixelShuffle(2x2x) → Conv(3) → Output
```

### 3. **ESRGAN** (Wang et al., 2018)

**Referência:** [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

- **Conceito:** GAN para super-resolução perceptual; foco em realismo visual
- **Arquitetura:** Gerador com blocos RRDB (Residual Dense Blocks) + Discriminador relativístico
- **Features:** Perceptual loss (VGG19), Adversarial loss, relaxamento relativístico
- **Vantagem:** Imagens mais realistas, menos suavização
- **Trade-off:** Pode alucinar detalhes; PSNR pode ser menor apesar de melhor qualidade visual

```
Gerador: Input → Conv(3) → [23 RRDB] → Conv(3) + Skip → Upsampling(2x2) → Conv(3) → Output
Discriminador: Input → [Conv layers with stride] → Dense layers → Realista/Fake
```

---

## 🔬 Metodologia Experimental

### Datasets

| Nome | Tipo | # Imagens | Uso | Detalhes |
|------|------|-----------|-----|----------|
| DIV2K | Treino | 800 (subset usado) | Treino | Imagens naturais de alta qualidade |
| Set5 | Benchmark | 5 | Avaliação | Muito fácil, convergência rápida |
| Set14 | Benchmark | 14 | Avaliação | Dificuldade média |
| BSD100 | Benchmark | 100 | Avaliação | Imagens naturais, gramas/texturas |
| Urban100 | Benchmark | 100 | Avaliação (extra) | Cenas urbanas, high-frequency |

### Protocolo de Avaliação

- **Fator de upscaling:** 4× (mais visualmente interessante, padrão nos papers)
- **Geração LR:** Downsampling bicúbico + degradação
- **Augmentação treino:** Flip horizontal + rotações 90° (padrão dos papers)
- **Métricas:**
  - **PSNR (Peak Signal-to-Noise Ratio):** Métrica pixel-a-pixel, tradicional mas correlação fraca com percepção
  - **SSIM (Structural Similarity Index):** Considera estrutura local, melhor que PSNR mas ainda não ideal
  - **Inspeção visual:** Grid comparativo de recortes ampliados (o que realmente importa)

### Hiperparâmetros

| Parâmetro | SRCNN | EDSR-baseline | ESRGAN |
|-----------|-------|---------------|--------|
| Loss | MSE (L2) | L1 | Perceptual + Adversarial |
| Optimizer | Adam | Adam | Adam (Gen + Disc) |
| Learning Rate | 0.001 | 0.001 | 0.001 |
| Batch Size | 16-32 | 16-32 | 4-8 (memória) |
| Epochs | ~50-100 | ~100-200 | ~100-200 |
| Scheduler | StepLR | CosineAnnealingLR | StepLR |
| Early Stopping | Patience=10 | Patience=15 | Não usado (GAN) |

---

## 🛠️ Decisões de Design Importantes

### 1. **Input: Bicúbic Pré-upscalado vs. LR Puro**
- **SRCNN:** Recebe LR já upscalado via bicúbica → funciona como "refinador"
- **EDSR/ESRGAN:** Recebem LR "pequeno" → aprendem upsampling do zero
- **Razão:** Papers originais usam essa abordagem; preservamos fidelidade histórica

### 2. **Loss Function: MSE vs. L1 vs. Perceptual**
- **MSE (SRCNN):** Simples, converge rápido, mas produz imagens suavizadas
- **L1 (EDSR):** Melhor para detalhes finos, menos suavização (recomendado em práticas modernas)
- **Perceptual + Adversarial (ESRGAN):** Força o gerador a "enganar" discriminador; realismo > fidelidade pixel-a-pixel

### 3. **Upsampling: PixelShuffle vs. Nearest + Conv**
- **PixelShuffle (EDSR):** Rearranjo inteligente de canais, sem checkerboard artifacts
- **Nearest + Conv (ESRGAN):** Mais controle, compatível com GAN discriminator

### 4. **Sem BatchNorm na EDSR-baseline**
- **Razão:** BatchNorm adiciona dependência do batch size; durante inference não há "batch", causando domain shift
- **Trade-off:** Convergência mais lenta, mas melhor generalização

### 5. **VGG19 Feature Extractor (ESRGAN)**
- Usa features pré-treinadas de ImageNet (camadas 35 de VGG19)
- Força o gerador a produzir imagens que ativam features "naturais" na rede
- Congelado (no gradients) para estabilidade

---

## 📁 Estrutura do Projeto

```
upscaling-ml/
├── 00_guia_analise_comparacao.ipynb    # 📖 COMECE AQUI - Análise exploratória e comparação
├── 01_train_srcnn.ipynb                # Treino do SRCNN (1h)
├── 02_train_edsr.ipynb                 # Treino do EDSR-baseline (3h)
├── 03_train_esrgan.ipynb               # Treino do ESRGAN (12h+)
├── models.py                           # Implementação das 3 arquiteturas
├── datasets.py                         # Carregamento de DIV2K, Set5, Set14, BSD100, Urban100
├── train.py                            # Loop de treino/avaliação genérico (fit, train_one_epoch, evaluate)
├── metrics.py                          # PSNR, SSIM (versão rápida + skimage)
├── edsrgan_model.ipynb                 # Notebook histórico de exploração
├── datasets/                           # Dados (DIV2K, Set5, Set14, BSD100, Urban100)
├── outputs/                            # Checkpoints e históricos de treino (experimentos)
└── README.md                           # Este arquivo
```

---

## 📖 Guia Prático: Como Usar os Notebooks

### 🚀 Início Rápido (5 min)

1. **Abra `00_guia_analise_comparacao.ipynb`**
   - Veja análise exploratória dos datasets
   - Entenda a estrutura dos dados
   - Visualize exemplos de patches

2. **Depois de treinar os modelos** (veja próximas seções)
   - Execute as células finais de `00_guia_analise_comparacao.ipynb`
   - Compare resultados dos 3 modelos
   - Gere tabelas e gráficos de comparação

---

### 🎯 Para Treinar os Modelos

**Opção A: Treinar um por um (recomendado)**

1. **SRCNN** (~1 hora em GPU)
   - Abra `01_train_srcnn.ipynb`
   - Execute todas as células
   - Modelo salvo em `outputs/SRCNN_2026-07-05/srcnn_final.pt`

2. **EDSR-Baseline** (~3 horas em GPU)
   - Abra `02_train_edsr.ipynb`
   - Execute todas as células
   - Modelo salvo em `outputs/EDSR_2026-07-05/edsr_final.pt`

3. **ESRGAN** (~12 horas em GPU)
   - Abra `03_train_esrgan.ipynb`
   - Execute todas as células
   - Modelos salvos em `outputs/ESRGAN_2026-07-05/`
   - ⏱️ **Dica:** Deixe rodando à noite

**Opção B: Usar modelos pré-treinados**
- Se os modelos já existem em `outputs/`, pule direto para `00_guia_analise_comparacao.ipynb`

---

### 🔧 Customizações Comuns

**Reduzir tempo de treino (teste rápido):**
```python
n_epochs = 10  # Em vez de 100
BATCH_SIZE = 8  # Em vez de 16
```

**Usar GPU diferente ou CPU:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Monitorar treino em tempo real:**
```bash
# Em outro terminal:
tensorboard --logdir outputs
# Acesse: http://localhost:6006
```

---

### 📊 Comparação de Resultados

Após treinar os 3 modelos, execute `00_guia_analise_comparacao.ipynb` para:
- ✅ Tabela comparativa (PSNR, SSIM em Set5, Set14, BSD100, Urban100)
- ✅ Visualização lado-a-lado (HR original, Bicubic, SRCNN, EDSR, ESRGAN)
- ✅ Zoom em detalhes de texturas
- ✅ Mapas de erro absoluto

---

---

## 🚀 Como Reproduzir

### 1. Instalação

```bash
# Criar ambiente virtual (recomendado)
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

**Nota:** `requirements.txt` detecta automaticamente plataforma (macOS usa MPS, Linux/Windows usam CUDA 12.8).

### 2. Preparar Dados

```bash
# Estrutura esperada:
# datasets/
# ├── DIV2K/
# │   ├── DIV2K_train_HR/
# │   └── DIV2K_valid_HR/
# ├── Set5/image_SRF_4/
# ├── Set14/image_SRF_4/
# ├── BSD100/image_SRF_4/
# └── Urban100/image_SRF_4/
```

Datasets precisam estar no formato padrão dos benchmarks (pares `*_HR.png` e `*_LR.png`).

### 3. Treinar um Modelo

```python
from models import SRCNN, EDSRBaseline, ESRGANGenerator
from datasets import SRDataset, SRBenchmarkDataset
from train import fit
import torch
from torch.utils.data import DataLoader

# Exemplo: treinar EDSR-baseline
train_data = SRDataset("datasets/DIV2K/DIV2K_train_HR", patch_size=96, scale=4)
val_data = SRBenchmarkDataset("datasets/Set5/image_SRF_4", scale=4)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1)

model = EDSRBaseline(num_features=64, num_blocks=16, scale=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

history = fit(
    model, train_loader, val_loader, optimizer, criterion, device,
    epochs=100, upscale_input=False, tag="edsr_experiment",
    patience=15, scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
)
```

Ver `edsrgan_model.ipynb` ou `upscaling(v7).ipynb` para exemplos completos.

### 4. Avaliar em Benchmarks

```python
from metrics import calc_psnr, calc_ssim, calc_ssim_skimage

# Carregar modelo treinado e processar benchmark
model.eval()
psnr_scores, ssim_scores = [], []

for lr, hr in val_loader:
    with torch.no_grad():
        sr = model(lr.to(device))
    psnr_scores.append(calc_psnr(sr, hr))
    ssim_scores.append(calc_ssim_skimage(sr, hr))

print(f"PSNR: {sum(psnr_scores)/len(psnr_scores):.2f}")
print(f"SSIM: {sum(ssim_scores)/len(ssim_scores):.4f}")
```

---

## 📊 Resultados

### Sumário de Performance (Fator 4×)

| Modelo | Set5 PSNR | Set5 SSIM | Set14 PSNR | BSD100 PSNR | Urban100 PSNR |
|--------|-----------|----------|-----------|-------------|---------------|
| Bicubic (baseline) | — | — | — | — | — |
| SRCNN | — | — | — | — | — |
| EDSR-baseline | — | — | — | — | — |
| ESRGAN | — | — | — | — | — |

**Status:** Resultados a serem preenchidos após consolidação dos experimentos em `outputs/` e `resultados_esrgan/`.

### Observações Qualitativas

> **Placeholder para análise visual.** Após treino final:
> 
> - **SRCNN:** Rápido, mas artefatos de ringing em bordas
> - **EDSR-baseline:** Ganho significativo em texturas finas vs. SRCNN
> - **ESRGAN:** Mais realista, menos suavização; pode alucinar detalhes em regiões ambíguas

---

## 🔍 Discussão: Fidelidade vs. Percepção

Um dos objetivos principais deste trabalho é destacar o **desacoplamento entre PSNR/SSIM e qualidade perceptual**.

### Observações esperadas

1. **SRCNN com PSNR alto != qualidade visual boa**
   - PSNR penaliza qualquer desvio pixel-a-pixel
   - Pequenas translações causam queda de PSNR desproporcional

2. **EDSR-baseline: ganho de detalhes**
   - Melhor em texturas finas (grama, tecido)
   - Menos suavização que SRCNN

3. **ESRGAN: realismo vs. fidelidade**
   - Pode ter PSNR ligeiramente menor
   - Visualmente mais atraente (menos artefatos, texturas mais naturais)
   - Risco: alucinação de detalhes inventados

### Métrica Alternativa

A literatura recente propõe métricas perceptuais (LPIPS, FID) que correlacionam melhor com percepção humana. Este projeto usa PSNR/SSIM por ser padrão, mas reconhece a limitação.

---

## 🎓 Aprendizados Principais

1. **Contexto importa:** A mesma arquitetura treina diferente em DIV2K vs. Set5
2. **Inicialização:** Xavier/He initialization é crítica para convergência
3. **Loss é tudo:** L1 > MSE para details; perceptual loss necessário para realismo
4. **Trade-off inerente:** Não há "melhor" modelo, apenas melhor para cada caso de uso

---

## 📝 Como Citar

Se usar este trabalho como referência, cite o formato:

```bibtex
@misc{abilio2026sr,
  author = {Abílio, Marcos and Vieira, Gabriel},
  title = {Super-Resolução de Imagens: Comparação de Arquiteturas},
  school = {Universidade Federal de Minas Gerais},
  year = {2026},
  note = {Projeto Final, Disciplina ML2}
}
```

---

## 📖 Referências Principais

1. Dong, C., Lim, B., Yu, K., & others. (2016). *Learning a Deep Convolutional Network for Image Super-Resolution.* ECCV.
2. Lim, B., Son, S., Kim, H., Nah, S., & Lee, K. M. (2017). *Enhanced Deep Residual Networks for Single Image Super-Resolution.* CVPR.
3. Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., & Change Loy, C. (2018). *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.* ECCV.
4. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR (LPIPS metric).

---

## 📞 Contato

Para dúvidas sobre implementação ou resultados:
- **Marcos Abílio:** [GitHub](https://github.com/ma-melo)
- **Gabriel Vieira:** [GitHub](https://github.com/gabrielgszv)

---

*Última atualização: julho de 2026*
