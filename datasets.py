"""
Datasets para o pipeline de super-resolução.

SRDataset         -> treino: crop aleatório + augmentação (flip + rotação)
SREvalDataset      -> avaliação simples: imagem inteira, LR gerado via bicubic
                       (usar quando não há pares LR/HR oficiais prontos)
SRBenchmarkDataset -> avaliação com benchmarks padrão (Set5, Set14, BSD100,
                       Urban100) no formato "image_SRF_<scale>", que já vem
                       com pares HR/LR prontos -- usar este para gerar
                       números comparáveis com os papers de referência.
"""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SRDataset(Dataset):
    """Dataset de treino: extrai patches HR aleatórios e gera o LR correspondente."""

    def __init__(self, folder, patch_size=96, scale=4):
        self.files = sorted(Path(folder).glob("*.png"))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"Nenhum .png encontrado em '{folder}'. Verifique o caminho."
            )
        self.patch_size = patch_size
        self.scale = scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size

        # Guard: imagens menores que patch_size quebrariam o randint.
        if w < self.patch_size or h < self.patch_size:
            img = img.resize(
                (max(w, self.patch_size), max(h, self.patch_size)),
                Image.BICUBIC,
            )
            w, h = img.size

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        hr = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # augmentação: flip horizontal + rotações de 90°
        if random.random() < 0.5:
            hr = TF.hflip(hr)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            hr = hr.rotate(angle)

        lr = hr.resize(
            (self.patch_size // self.scale, self.patch_size // self.scale),
            Image.BICUBIC,
        )

        return TF.to_tensor(lr), TF.to_tensor(hr)


class SREvalDataset(Dataset):
    """Dataset de avaliação simples: usa a imagem inteira, gera o LR via bicubic.

    Use esta classe quando você só tem imagens HR soltas (sem pares LR
    oficiais prontos) -- por exemplo DIV2K_valid_HR sem subpastas image_SRF_*.
    """

    def __init__(self, folder, scale=4):
        self.files = sorted(Path(folder).glob("*.png"))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"Nenhum .png encontrado em '{folder}'. Verifique o caminho "
                f"(ex: pasta extraída corretamente)."
            )
        self.scale = scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size

        # garante que w,h sejam múltiplos do scale (evita erro de dimensão)
        w = w - (w % self.scale)
        h = h - (h % self.scale)
        hr = img.crop((0, 0, w, h))

        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)

        return TF.to_tensor(lr), TF.to_tensor(hr)


class SRBenchmarkDataset(Dataset):
    """Lê pares HR/LR já prontos de um benchmark (Set5, Set14, BSD100, Urban100).

    Espera uma estrutura como:
        <folder>/img_001_SRF_4_HR.png
        <folder>/img_001_SRF_4_LR.png
        <folder>/img_002_SRF_4_HR.png
        <folder>/img_002_SRF_4_LR.png
        ...

    Diferente do SREvalDataset (que gera o LR na hora via bicubic), aqui
    usamos o LR oficial do benchmark -- o que torna os resultados
    comparáveis com os papers de referência (Dong, Lim, Wang).

    Parâmetros
    ----------
    folder : str | Path
        Caminho para a pasta "image_SRF_<scale>" (ex: "datasets/Set5/image_SRF_4")
    scale : int
        Fator de escala esperado. É validado contra a razão real entre
        LR e HR; se não corresponder, um aviso é emitido (provável pasta
        errada, ex: passou scale=4 mas a pasta é image_SRF_3).
    """

    def __init__(self, folder, scale=4):
        folder = Path(folder)
        hr_files = sorted(folder.glob("*_HR.png"))

        if len(hr_files) == 0:
            raise FileNotFoundError(
                f"Nenhum arquivo '*_HR.png' encontrado em '{folder}'. "
                f"Confirme o caminho (ex: 'datasets/Set5/image_SRF_{scale}')."
            )

        self.pairs = []
        for hr_path in hr_files:
            # opera só no nome do arquivo (não no caminho completo), para
            # evitar substituições indesejadas caso o caminho contenha "_HR.png"
            lr_name = hr_path.name.replace("_HR.png", "_LR.png")
            lr_path = hr_path.with_name(lr_name)

            if not lr_path.exists():
                raise FileNotFoundError(
                    f"Par LR não encontrado para '{hr_path.name}' "
                    f"(esperado: '{lr_path.name}')."
                )
            self.pairs.append((lr_path, hr_path))

        self.scale = scale
        self._scale_checked = False

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]

        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        lw, lh = lr.size
        hw, hh = hr.size

        # Validação: a razão real HR/LR deve corresponder ao scale informado.
        # Só checa uma vez (no primeiro item) para não gerar warnings repetidos.
        if not self._scale_checked:
            real_ratio_w = hw / lw
            real_ratio_h = hh / lh
            if abs(real_ratio_w - self.scale) > 0.5 or abs(real_ratio_h - self.scale) > 0.5:
                print(
                    f"[AVISO] scale={self.scale} informado, mas a razão real "
                    f"HR/LR é ~{real_ratio_w:.2f}x. Confirme se a pasta "
                    f"'{lr_path.parent}' corresponde ao fator de escala correto."
                )
            self._scale_checked = True

        expected_hw, expected_hh = lw * self.scale, lh * self.scale
        if (hw, hh) != (expected_hw, expected_hh):
            hr = hr.crop((0, 0, expected_hw, expected_hh))

        return TF.to_tensor(lr), TF.to_tensor(hr)