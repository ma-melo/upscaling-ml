# Projeto Final da Disciplina de [ML2](https://ganacim.github.io/ml2-2026/)
## Problema de Super-Resolução de Imagens
### Alunos: [Marcos Abílio](https://github.com/ma-melo) e [Gabriel Vieira](https://github.com/gabrielgszv)
 
## 1. Objetivo do trabalho
 
Treinar e comparar arquiteturas de super-resolução (SR) de imagens em datasets
padrão, avaliando o compromisso entre métricas tradicionais (PSNR, SSIM) e
qualidade perceptual (inspeção visual de texturas e artefatos).
 
Modelos de referência: SRCNN (Dong et al., 2016), EDSR (Lim et al., 2017) e
ESRGAN (Wang et al., 2018).

---
 
## 2. Plano em fases
 
### Fase 1 — Baseline mínimo (SRCNN)
**Meta:** ter um pipeline completo funcionando ponta a ponta, mesmo que simples.
 
- Implementar o pipeline de dados (carregar imagens, gerar pares LR/HR)
- Implementar a SRCNN (3 camadas conv)
- Treinar em subconjunto pequeno de DIV2K
- Avaliar em Set5 com PSNR/SSIM
- Comparar contra bicubic puro (baseline "sem rede")
**Critério de sucesso:** SRCNN supera bicubic em PSNR/SSIM no Set5.
 
### Fase 2 — Modelo residual leve (EDSR-baseline)
**Meta:** comparar arquitetura residual sem BN contra a SRCNN.
 
- Implementar EDSR-baseline (16 blocos residuais, 64 filtros, sem BatchNorm)
- Mesmo pipeline de dados, agora treinando direto em RGB (sem upscale bicubic prévio — a rede aprende o upsampling)
- Treinar com loss L1
- Avaliar nos mesmos datasets (Set5, Set14, BSD100)
- Comparar contra SRCNN e bicubic
**Critério de sucesso:** EDSR-baseline supera SRCNN em PSNR/SSIM.
 
### Fase 3 — Análise e discussão (obrigatória)
- Tabela comparativa final (bicubic / SRCNN / EDSR-baseline) em todos os datasets de teste
- Grid de imagens lado a lado (recortes ampliados, tipo "zoom" em região com textura fina)
- Discussão: onde o ganho de PSNR não corresponde a ganho visual perceptível? Onde os artefatos aparecem (bordas, texturas repetitivas tipo grama/tijolo)?
### Fase 4 — Bônus (se houver tempo/recursos)
Escolher **um ou dois** itens, não todos:
- Adicionar perceptual loss (VGG) ao EDSR-baseline e comparar visualmente vs. loss L1 puro
- Implementar versão simplificada do discriminador relativístico do ESRGAN (RaGAN) treinando um GAN pequeno
- Testar outro tipo de degradação além de bicubic (ex: blur gaussiano + downsample, ou ruído)
- Comparar custo computacional (tempo de inferência, nº de parâmetros) entre os modelos
---
 
## 3. Detalhes técnicos do pipeline
 
### 3.1 Dados
- **Treino:** DIV2K (ou subconjunto, ex: 200–400 imagens, dependendo do poder computacional disponível)
- **Teste:** Set5, Set14, BSD100 (Urban100 como extra, é mais desafiador)
- **Fator de upscaling:** escolher um único fator para focar (recomendado: ×4, é o mais usado nos três papers e o mais visualmente interessante)
### 3.2 Geração dos pares LR/HR
1. Recortar patches HR da imagem original (ex: 96×96 para EDSR, 33×33 para SRCNN)
2. Aplicar downsampling bicubic pelo fator de escala → gera o LR
3. Augmentação: flip horizontal + rotações de 90° (padrão dos três papers)
4. Para SRCNN: upscalar o LR de volta ao tamanho HR via bicubic antes de entrar na rede
5. Para EDSR: o LR entra "pequeno" mesmo, a rede faz o upsampling internamente
### 3.3 Treinamento
| | SRCNN | EDSR-baseline |
|---|---|---|
| Loss | MSE (L2) | L1 |
| Otimizador | Adam ou SGD+momentum | Adam |
| Canal | Y (luminância, YCbCr) | RGB |
| Batch size | conforme memória disponível | conforme memória disponível |
 
### 3.4 Avaliação
- PSNR e SSIM calculados no **canal Y** (padrão da literatura, mesmo se o modelo treinou em RGB — converter a saída para YCbCr antes de medir)
- Ignorar uma borda de pixels equivalente ao fator de escala (prática comum nos papers, evita penalizar artefatos de borda)
- Reportar média por dataset de teste
---
 
## 4. Estrutura sugerida do relatório final
 
1. Introdução ao problema de SR e formalização do problema mal-posto
2. Descrição dos modelos implementados (arquitetura, loss, diferenças)
3. Metodologia experimental (datasets, fator de escala, métricas, hiperparâmetros)
4. Resultados quantitativos (tabela PSNR/SSIM por dataset)
5. Resultados qualitativos (imagens comparativas, zooms em detalhes)
6. Discussão: tradeoff fidelidade vs. percepção, limitações do PSNR/SSIM
7. (Se houver) Resultados do item bônus
8. Conclusão
---
 
## 5. Próximo passo imediato
 
Decidir o ambiente de execução (GPU disponível? Google Colab? CPU local?) — isso
define o tamanho do subconjunto de DIV2K e dos modelos que é viável treinar
dentro do tempo do trabalho.
 
