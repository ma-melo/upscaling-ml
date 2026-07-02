"""
Script completo para treinar e comparar SRCNN vs EDSR-baseline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime

from models import SRCNN, EDSRBaseline
from datasets import SRDataset, SREvalDataset, SRBenchmarkDataset
from train import fit, evaluate

# Configuração
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {DEVICE}")

DIV2K_TRAIN = Path("datasets/DIV2K_train_HR")
DIV2K_VALID = Path("datasets/DIV2K_valid_HR")
SET5 = Path("datasets/Set5/image_SRF_4")
SET14 = Path("datasets/Set14/image_SRF_4")
BSD100 = Path("datasets/BSD100/image_SRF_4")
URBAN100 = Path("datasets/Urban100/image_SRF_4")

SCALE = 4
BATCH_SIZE = 16
NUM_EPOCHS = 50
PATCH_SIZE = 96
LR = 1e-4

# Criar datasets
print("\n[1/4] Carregando datasets...")
train_dataset = SRDataset(DIV2K_TRAIN, patch_size=PATCH_SIZE, scale=SCALE)
val_dataset = SREvalDataset(DIV2K_VALID, scale=SCALE)

benchmarks = {
    "Set5": SRBenchmarkDataset(SET5, scale=SCALE),
    "Set14": SRBenchmarkDataset(SET14, scale=SCALE),
    "BSD100": SRBenchmarkDataset(BSD100, scale=SCALE),
    "Urban100": SRBenchmarkDataset(URBAN100, scale=SCALE),
}

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
benchmark_loaders = {
    name: DataLoader(ds, batch_size=1, num_workers=4)
    for name, ds in benchmarks.items()
}

print(f"  Train: {len(train_dataset)} imagens")
print(f"  Val: {len(val_dataset)} imagens")
print(f"  Benchmarks: {', '.join(benchmarks.keys())}")

# ==============================================================================
# Treino SRCNN
# ==============================================================================
print("\n[2/4] Treinando SRCNN...")
srcnn = SRCNN().to(DEVICE)
srcnn_optimizer = optim.Adam(srcnn.parameters(), lr=LR)
srcnn_criterion = nn.MSELoss()

# Learning rate scheduler: decay de 0.5x a cada 30 épocas
srcnn_scheduler = optim.lr_scheduler.StepLR(srcnn_optimizer, step_size=30, gamma=0.5)

srcnn_history = fit(
    srcnn, train_loader, val_loader, srcnn_optimizer, srcnn_criterion, DEVICE,
    epochs=NUM_EPOCHS, scale=SCALE, upscale_input=True, scheduler=srcnn_scheduler
)

# Avaliar SRCNN nos benchmarks
print("\nAvaliando SRCNN nos benchmarks...")
srcnn_results = {}
for name, loader in benchmark_loaders.items():
    metrics = evaluate(srcnn, loader, DEVICE, scale=SCALE, upscale_input=True)
    srcnn_results[name] = metrics
    print(f"  {name}: PSNR={metrics['psnr_model']:.2f} | SSIM={metrics['ssim_model']:.4f}")

# ==============================================================================
# Treino EDSR-baseline
# ==============================================================================
print("\n[3/4] Treinando EDSR-baseline...")
edsr = EDSRBaseline(num_features=64, num_blocks=16, scale=SCALE).to(DEVICE)
edsr_optimizer = optim.Adam(edsr.parameters(), lr=LR)
edsr_criterion = nn.L1Loss()

# Learning rate scheduler: decay de 0.5x a cada 30 épocas
edsr_scheduler = optim.lr_scheduler.StepLR(edsr_optimizer, step_size=30, gamma=0.5)

edsr_history = fit(
    edsr, train_loader, val_loader, edsr_optimizer, edsr_criterion, DEVICE,
    epochs=NUM_EPOCHS, scale=SCALE, upscale_input=False, scheduler=edsr_scheduler
)

# Avaliar EDSR nos benchmarks
print("\nAvaliando EDSR-baseline nos benchmarks...")
edsr_results = {}
for name, loader in benchmark_loaders.items():
    metrics = evaluate(edsr, loader, DEVICE, scale=SCALE, upscale_input=False)
    edsr_results[name] = metrics
    print(f"  {name}: PSNR={metrics['psnr_model']:.2f} | SSIM={metrics['ssim_model']:.4f}")

# ==============================================================================
# Salvar resultados
# ==============================================================================
print("\n[4/4] Salvando resultados...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("outputs") / f"results_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)

results = {
    "timestamp": timestamp,
    "device": str(DEVICE),
    "config": {
        "scale": SCALE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "patch_size": PATCH_SIZE,
        "lr": LR,
    },
    "srcnn": {
        "history": srcnn_history,
        "benchmarks": srcnn_results,
    },
    "edsr": {
        "history": edsr_history,
        "benchmarks": edsr_results,
    },
}

# Salvar JSON com resultados
with open(results_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Salvar checkpoints dos modelos
torch.save(srcnn.state_dict(), results_dir / "srcnn_final.pt")
torch.save(edsr.state_dict(), results_dir / "edsr_final.pt")

print(f"\n✅ Resultados salvos em: {results_dir}")
print(f"  - results.json (métricas completas)")
print(f"  - srcnn_final.pt (pesos do modelo)")
print(f"  - edsr_final.pt (pesos do modelo)")

# ==============================================================================
# Resumo comparativo
# ==============================================================================
print("\n" + "="*80)
print("RESUMO COMPARATIVO")
print("="*80)

for benchmark_name in benchmarks.keys():
    print(f"\n{benchmark_name}:")
    print(f"  SRCNN | PSNR: {srcnn_results[benchmark_name]['psnr_model']:.2f} dB | "
          f"SSIM: {srcnn_results[benchmark_name]['ssim_model']:.4f}")
    print(f"  EDSR  | PSNR: {edsr_results[benchmark_name]['psnr_model']:.2f} dB | "
          f"SSIM: {edsr_results[benchmark_name]['ssim_model']:.4f}")

    psnr_diff = edsr_results[benchmark_name]['psnr_model'] - srcnn_results[benchmark_name]['psnr_model']
    ssim_diff = edsr_results[benchmark_name]['ssim_model'] - srcnn_results[benchmark_name]['ssim_model']

    winner = "EDSR ✓" if psnr_diff > 0 else "SRCNN ✓" if psnr_diff < 0 else "Empate"
    print(f"  Diferença: PSNR {psnr_diff:+.2f} dB | SSIM {ssim_diff:+.4f} | {winner}")
