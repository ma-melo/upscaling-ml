# Learning Rate Strategies para Super-Resolução

## Resumo do Progresso

| Métrica | v2 (20 épocas) | v3 (100 épocas) | Melhoria |
|---------|---|---|---|
| SRCNN Set5 PSNR | 25.45 | 27.01 | +1.56 dB ✅ |
| Supera Bicubic? | ❌ Não | ✅ Sim | Crítico! |

**Conclusão:** O problema inicial não era learning rate, mas sim **número de épocas**. A rede precisava de mais tempo para convergir adequadamente.

---

## 1. Learning Rate Scheduler: O que é?

Um **scheduler** ajusta o learning rate durante o treinamento. Estratégia comum:
- **Primeiras épocas:** LR alto (aprendizado rápido)
- **Depois:** LR decai progressivamente (fine-tuning preciso)

### Por que não simplesmente usar LR mais alto desde o início?

```
LR alto demais → Oscilações, divergência, treinamento instável
LR baixo demais → Convergência lenta ou presa em mínimo local
```

---

## 2. Schedulers Implementados no train-models.py

### StepLR (Decay por passo)

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
# A cada 30 épocas, multiplica LR por 0.5
# Épocas 0-29: LR = 1e-4
# Épocas 30-59: LR = 5e-5
# Épocas 60-89: LR = 2.5e-5
# Épocas 90+: LR = 1.25e-5
```

**Vantagem:** Simples, previsível
**Desvantagem:** Mudanças abruptas; pode não ser ótimo se a rede ainda está aprendendo bem

---

## 3. Alternativas (para exploração futura)

### a) MultiStepLR (Decay em milestones específicos)

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[50, 80],  # Decay em épocas 50 e 80
    gamma=0.5
)
# Épocas 0-49: LR = 1e-4
# Épocas 50-79: LR = 5e-5
# Épocas 80+: LR = 2.5e-5
```

**Melhor para:** Ajuste fino de quando aplicar decay

### b) CosineAnnealingLR (Redução suave)

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # Número total de épocas
    eta_min=1e-6  # LR mínimo
)
# LR segue uma curva cossenoidal, decaindo suavemente
# Menos abrupto que StepLR
```

**Melhor para:** Treinamento prolongado com fine-tuning gradual

### c) ExponentialLR (Decay exponencial)

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# A cada época: LR *= 0.95
# Decay suave e contínuo
```

---

## 4. Como Adicionar ao Seu Código

**No train-models.py (já adicionado):**

```python
# Criar scheduler
srcnn_scheduler = optim.lr_scheduler.StepLR(srcnn_optimizer, step_size=30, gamma=0.5)

# Passar para fit()
srcnn_history = fit(
    srcnn, train_loader, val_loader, srcnn_optimizer, srcnn_criterion, DEVICE,
    epochs=NUM_EPOCHS, scale=SCALE, upscale_input=True,
    scheduler=srcnn_scheduler  # ← Adicione isto
)
```

**No notebook (projeto(v3).ipynb):**

```python
# Para SRCNN
srcnn_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

for epoch in range(n_epochs):
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    srcnn_scheduler.step()  # ← Adicione isto após cada época
    # ... resto do código
```

---

## 5. Recomendações para SR

### Para SRCNN (arquitetura simples)
- **Sugestão:** `StepLR(step_size=30, gamma=0.5)`
- **Por quê:** SRCNN é simples, decay moderado funciona bem

### Para EDSR-baseline (arquitetura maior)
- **Sugestão:** `MultiStepLR(milestones=[50, 80], gamma=0.5)`
- **Por quê:** Mais flexibilidade; EDSR pode continuar aprendendo alem da época 30

### Para ambos (estratégia "segura")
- **Sugestão:** `CosineAnnealingLR(T_max=100, eta_min=1e-6)`
- **Por quê:** Sem cortes abruptos; simula naturalmente o "resfriamento" do aprendizado

---

## 6. Como Monitorar

Se usar TensorBoard, adicione isso ao loop:

```python
writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
```

Visualize no TensorBoard para ver o decaimento do LR em tempo real.

---

## 7. Resumo: O que o Seu Projeto Precisa Fazer Agora

✅ **Já feito:** Aumentar épocas de 20 → 100 (v3)  
✅ **Já feito:** Learning rate scheduler adicionado a train-models.py  

### Próximos passos:
1. **(Opcional)** Testar com scheduler no train-models.py
2. **(Obrigatório)** Finalizar análise qualitativa (Fase 3)
3. **(Bônus)** Se houver tempo, implementar perceptual loss ou compare computational cost

---

## Referências

- PyTorch LR Schedulers: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- EDSR paper (Lim et al., 2017): Usa decay linear decadente
- SRCNN paper (Dong et al., 2016): Usa SGD+momentum fixo

