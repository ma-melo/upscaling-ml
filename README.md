# Testando Arquiteturas para Super-Resolução de Imagens

**Projeto Final da Disciplina [ML2](https://ganacim.github.io/ml2-2026/)** (2026)

**Alunos:** [Marcos Abílio](https://github.com/ma-melo) e [Gabriel Vieira](https://github.com/gabrielgszv)

**Repositório:** https://github.com/ma-melo/upscaling-ml

---

## Introdução

Este trabalho implementa, treina e compara três arquiteturas clássicas de super-resolução (SR) de imagens: **SRCNN**, **EDSR-baseline** e **ESRGAN**. O objetivo é avaliar o trade-off entre métricas tradicionais (PSNR, SSIM) e qualidade perceptual.

---

## O que é Super-Resolução de Imagens?

A super-resolução é o problema de aumentar a resolução de uma imagem de baixa resolução (LR) para alta resolução (HR), reconstruindo detalhes que foram perdidos durante a degradação. É um problema **mal-posto** (ill-posed) porque múltiplas imagens HR podem produzir a mesma imagem LR.

### Aplicações práticas
- Melhoramento de câmeras de vigilância
- Aumento de resolução de imagens históricas/médicas
- Preparação de conteúdo para exibição em telas maiores

### Desafio técnico
O espaço de possíveis soluções é enorme. Modelos ingênuos (interpolação bicúbica) produzem imagens borradas. Redes neurais aprendem a alucinar detalhes realistas, mas com trade-offs:
- **Fidelidade pixel-a-pixel** (PSNR alto) vs. **Qualidade perceptual** (menos artefatos, texturas mais naturais)

---

## Arquiteturas de Referência

### **SRCNN**

**Referência:** [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)

- **Arquitetura:** 3 camadas convolucionais. A primeira extrai feature maps da imagem de baixa resolução, a segunda mapeia essas características para o patch de alta resolução e a última combina tudo para gerar a imagem final.
- **Entrada:** Imagem LR já upscalada para tamanho HR via bicúbica
- **Vantagem:** Rápida e de baixo uso de memória
- **Limitação:** Sem residuais e dificuldade em treinar profundidade

```
Input (bicúbic upscaled) → Conv(9) → ReLU → Conv(1) → ReLU → Conv(5) → Output
```

### 2. **EDSR**

**Referência:** [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)

- **Arquitetura:** 16 blocos residuais (64 features), sem BatchNorm, upsampling via PixelShuffle
- **Entrada:** Imagem LR "pequena" e a rede aprende o upsampling internamente
- **Vantagem:** Melhor performance (PSNR) e treino mais estável

```
Input (LR) → Conv(3) → [16 ResidualBlocks] → Conv(3) + Skip → PixelShuffle(2x2x) → Conv(3) → Output
```

### 3. **ESRGAN**

**Referência:** [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

Mesmo com os modelos anteriores apresentando boa avaliação nas métricas clássicas (como PSNR), a dependência da otimização do erro absoluto (L1) restringe o modelo na formação de texturas mais complexas e naturais. O ESRGAN melhora a arquitetura das GANs através de três modificações:

- **Arquitetura do Gerador:** os blocos residuais convencionais foram substituídos por blocos RRDB (Residual-in-Residual Dense Block). Esse bloco mantém a ideia do EDSR de não usar Batch Normalization e introduz conexões mais densas dentro dos próprios blocos residuais, permitindo que a rede recupere características complexas desde as camadas iniciais até o final do processamento, o que garante maior estabilidade no aprendizado de texturas.
- **Discriminador Relativístico Médio (RaD):** ao contrário do discriminador padrão de uma GAN (que estima se uma imagem é falsa ou real), o ESRGAN usa o *Relativistic average Discriminator*, que calcula a probabilidade de uma imagem real ser "mais realista" que uma imagem gerada. Isso força o gerador a criar detalhes de forma mais agressiva para enganar o discriminador.
- **Perda Perceptual aprimorada:** a função de custo total combina a perda por pixel (L1), a perda adversarial (RaD) e a perda perceptual baseada na rede VGG. A VGG não atua como classificador, mas como avaliador de texturas, comparando os mapas de características extraídos **antes** da função de ativação, o que preserva mais informação sobre a textura.

Por fim, o ESRGAN faz uma **interpolação de redes**: os pesos de um modelo treinado só com perda L1 (garantindo estabilidade de cor e geometria) são interpolados com os pesos do modelo GAN (focado em realismo), equilibrando exatidão de pixel e realismo.

```
Gerador: Input → Conv(3) → [RRDB Blocks] → Conv(3) + Skip → Upsampling(2x2) → Conv(3) → Output
Discriminador: Input → [Conv layers with stride] → Dense layers → Realista/Fake
```

---

## Metodologia Experimental

### Datasets

| Nome | Tipo | # Imagens | Uso |
|------|------|-----------|-----|
| DIV2K_train | Treino | 800 | Treino do SRCNN, EDSR e ESRGAN |
| DF2K (DIV2K + Flickr2K) | Treino | 3.450 | Treino do ESRGAN |
| DIV2K_valid | Benchmark | 100 | Avaliação |
| Set5 | Benchmark | 5 | Avaliação |
| Set14 | Benchmark | 14 | Avaliação |
| BSD100 | Benchmark | 100 | Avaliação |
| Urban100 | Benchmark | 100 | Avaliação |

As imagens de baixa resolução foram obtidas aplicando degradação por subamostragem bicúbica com escala de **4×**. Foram usados patches com dimensões adaptadas a cada modelo, conforme as implementações originais dos papers de referência. Para evitar overfitting e aumentar a variedade do dataset, foi aplicado Data Augmentation com rotações aleatórias e espelhamentos.

### Protocolo de Treino por Modelo

- **SRCNN:** função de perda de Erro Quadrático Médio (**MSE**), minimizando a distância entre os pixels gerados e os de referência.
- **EDSR:** otimização baseada em Erro Absoluto Médio (**perda L1**), mais adequada para reconstrução de contornos e bordas. Sem camadas de Batch Normalization.
- **ESRGAN:** devido à instabilidade do treino de GANs, o treino foi separado em duas fases:
  - **Fase 1 (épocas 1–20):** o gerador é treinado minimizando apenas a perda L1, formando uma base geométrica.
  - **Fase 2 (épocas 21–150):** a função de custo passa a combinar perda L1, perda Adversarial e perda Perceptual (VGG).
  - Após o treino, os pesos das duas fases foram interpolados (**80% GAN + 20% fase 1**).

### Métricas de Avaliação

Os modelos foram comparados com a interpolação bicúbica nos conjuntos de teste **Set5** e **DIV2K**, usando:

- **PSNR (Peak Signal-to-Noise Ratio):** mede a fidelidade pixel a pixel com base no erro quadrático médio.
- **SSIM (Structural Similarity Index Measure):** avalia a preservação da estrutura com base em luminância, contraste e estrutura, variando de 0 a 1.
- **Inspeção visual:** grid comparativo de recortes ampliados, o que realmente importa para percepção humana.

---

## Decisões de Design Importantes

### 1. Input: Bicúbico Pré-upscalado vs. LR Puro
- **SRCNN:** recebe LR já upscalado via bicúbica → funciona como "refinador"
- **EDSR/ESRGAN:** recebem LR "pequeno" → aprendem o upsampling do zero
- **Razão:** os papers originais usam essa abordagem; preservamos a fidelidade histórica

### 2. Loss Function: MSE vs. L1 vs. Perceptual
- **MSE (SRCNN):** simples, converge rápido, mas produz imagens mais suavizadas
- **L1 (EDSR):** melhor para detalhes finos, menos suavização
- **Perceptual + Adversarial (ESRGAN):** força o gerador a "enganar" o discriminador; prioriza realismo sobre fidelidade pixel-a-pixel

### 3. Sem BatchNorm no EDSR
- **Razão:** BatchNorm adiciona dependência do tamanho do batch; durante a inferência não há "batch", causando domain shift
- **Trade-off:** convergência mais lenta, mas melhor generalização

### 4. Interpolação de Redes no ESRGAN
- Combina os pesos do modelo L1 (fase 1) com os do modelo GAN (fase 2) na proporção 20%/80%
- Estabiliza cores e reduz artefatos, equilibrando exatidão de pixel com realismo perceptual

---

## Estrutura do Projeto

```
upscaling-ml/
├── 00_guia_analise_comparacao.ipynb    # Análise exploratória e comparação entre os modelos treinados
├── 01_train_srcnn.ipynb           # Treino do SRCNN
├── 02_train_edsr.ipynb            # Treino do EDSR
├── 03_train_esrgan.ipynb          # Treino do ESRGAN
├── 04_train_esrgan_df2k.ipynb     # Treino do ESRGAN
├── models.py                      # Implementação das 3 arquiteturas
├── datasets.py                    # Carregamento de DIV2K, DF2K, Set5, Set14, BSD100, Urban100
├── train.py                       # Loop de treino/avaliação genérico (fit, train_one_epoch, evaluate)
├── metrics.py                     # PSNR, SSIM
├── datasets/                      # Datasets (DIV2K, DF2K, Set5, Set14, BSD100, Urban100)
├── outputs/                       # Checkpoints e históricos de treino (experimentos)
└── README.md                      # Este arquivo
```

---

## Guia Prático: Como Usar os Notebooks

### Início Rápido

1. **Abra `00_guia_analise_comparacao.ipynb`**
   - Veja a análise exploratória dos datasets
   - Entenda a estrutura dos dados
   - Visualize exemplos de patches

2. **Depois de treinar os modelos** (veja a seção abaixo)
   - Execute as células finais de `00_guia_analise_comparacao.ipynb`
   - Compare os resultados dos 3 modelos
   - Gere tabelas e gráficos de comparação

### Para Treinar os Modelos (Precisa baixar os datasets antes)

**Opção A: Treinar um por um**

1. **SRCNN**
   - Abra `01_train_srcnn.ipynb` e execute todas as células
   - Modelo salvo em `outputs/SRCNN_<data>/srcnn_final.pt`

2. **EDSR-Baseline**
   - Abra `02_train_edsr.ipynb` e execute todas as células
   - Modelo salvo em `outputs/EDSR_<data>/edsr_final.pt`

3. **ESRGAN**
   - Abra `03_train_esrgan.ipynb` e execute todas as células
   - Modelos salvos em `outputs/ESRGAN_<data>/` (variantes DIV2K e DF2K)

**Opção B (Recomendado): Usar modelos pré-treinados**
- Como os modelos já existem em `outputs/`, pule direto para `00_guia_analise_comparacao.ipynb`

### Comodidades

**Tensor Board:**
```bash
tensorboard --logdir outputs
# Acesse: http://localhost:6006
```

---

## Resultados

### Sumário de Performance (Fator 4×, médias em Set5 e DIV2K)

| Arquitetura | Função de Custo Base | PSNR (dB) | SSIM |
|---|---|---|---|
| Interpolação Bicúbica | — | 26.82 | 0.9953 |
| SRCNN | Perda L2 (MSE) | 25.56 | 0.9797 |
| EDSR | Perda L1 | **27.77** | 0.9753 |
| ESRGAN | Perda Perceptual + GAN | 23.47 | 0.9575 |

**Observações sobre os números:**
- O **EDSR** foi o único modelo a superar a interpolação bicúbica em PSNR (27.77 dB), confirmando o ganho esperado de uma arquitetura residual profunda treinada com perda L1.
- O **SRCNN** ficou abaixo do bicúbico (25.56 dB). Isso se deve à quantidade reduzida de épocas de treino usada para viabilizar a comparação entre arquiteturas (50 épocas em vez de 100); em testes à parte, a versão treinada por 100 épocas superou o bicúbico.
- O **ESRGAN** teve os menores valores de PSNR (23.47 dB) e SSIM (0.9575), o que é esperado, já que sua função de perda é dominada por penalizações adversariais e perceptuais, que otimizam para realismo visual e não para fidelidade pixel-a-pixel.
- De modo geral, tanto o SRCNN quanto o ESRGAN precisariam de mais épocas de treino para atingir os patamares reportados nos papers de referência.

### Exemplos Visuais

**Original**

![Original](imagens_saida/0810_original_HR.png)

**SRCNN**

![Resultados SRCNN](imagens_saida/0810_resultado_srcnn.png)

**EDSR**

![Resultados EDSR](imagens_saida/0810_resultado_edsr.png)

**ESRGAN**

![Resultados ESRGAN](imagens_saida/0810_resultado_esrgan.png)

**ESRGAN DF2K**

![Resultados ESRGAN DF2K](imagens_saida/0810_resultado_esrgan_df2k.png)

**Teste dos Modelos Principais no Set14**

![Modelos Principais](outputs/RESULTS_2026-07-08/visualizacao_resultados.png)

---

## Discussão: Fidelidade vs. Percepção

Um dos objetivos principais deste trabalho é destacar o **desacoplamento entre PSNR/SSIM e qualidade perceptual**.

1. **SRCNN e EDSR:** ambos suavizam alguns objetos e deixam a imagem pouco nítida, mesmo apresentando boas métricas de fidelidade pixel-a-pixel.
2. **ESRGAN:** mostrou o melhor resultado para a *visualização* da imagem. O processo de interpolação de redes se mostrou eficaz, gerando imagens sem ruídos e mais realistas, mesmo com PSNR/SSIM inferiores aos outros modelos.
3. **Conclusão prática:** PSNR e SSIM penalizam qualquer desvio pixel-a-pixel (inclusive pequenas translações), o que não necessariamente reflete qualidade percebida por um humano. O ESRGAN evidencia esse desacoplamento: métricas piores, resultado visual melhor.

### Métrica Alternativa

A literatura recente propõe métricas perceptuais (LPIPS, FID) que correlacionam melhor com a percepção humana. Este projeto usa PSNR/SSIM por serem o padrão da área, mas reconhece essa limitação.

---

## Aprendizados Principais

1. Treinar mais épocas: reduzir épocas para viabilizar comparações (SRCNN em 50 em vez de 100) tem impacto direto e mensurável no PSNR final.
2. A função de loss: L1 > MSE para detalhes finos; perda perceptual + adversarial é necessária para realismo, mesmo que penalize as métricas tradicionais.
3. Trade-off inerente: não existe melhor modelo isolado. EDSR ganha em fidelidade métrica, enquanto ESRGAN ganha em qualidade perceptual.
4. PSNR/SSIM têm limites claros: métricas altas não garantem boa percepção visual, e vice-versa.

---

## Referências

1. DONG, Chao; LOY, Chen Change; HE, Kaiming; TANG, Xiaoou. *Image Super-Resolution Using Deep Convolutional Networks*. IEEE Transactions on Pattern Analysis and Machine Intelligence. Disponível em: https://arxiv.org/abs/1501.00092. Acesso em: 13 junho 2026.
2. LIM, Bee; SON, Sanghyun; KIM, Heewon; NAH, Seungjun; MU LEE, Kyoung. *Enhanced Deep Residual Networks for Single Image Super-Resolution*. CVPR Workshops. Disponível em: https://arxiv.org/abs/1707.02921. Acesso em: 13 junho 2026.
3. WANG, Xintao; YU, Ke; WU, Shixiang; GU, Jinjin; LIU, Yihao; DONG, Chao; QIAO, Yu; LOY, Chen Change. *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*. ECCV Workshops. Disponível em: https://arxiv.org/abs/1809.00219. Acesso em: 13 junho 2026.

---

*Última atualização: julho de 2026*