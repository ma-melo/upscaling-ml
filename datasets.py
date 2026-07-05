# preparacao dos datasets para o pipeline de super-resolucao
#
# srdataset -> treino: crop aleatorio + augmentacao (flip + rotacao)
# srevaldataset -> avaliacao simples: imagem inteira, lr gerado via bicubic (usar quando nao ha pares lr/hr oficiais prontos)
# srbenchmarkdataset -> avaliacao com benchmarks padrao (set5, set14, bsd100, urban100) no formato "image_srf_<scale>", ja vem com pares hr/lr prontos

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


class SRDataset(Dataset):
    # dataset de treino: extrai patches hr aleatorios e gera o lr correspondente

    def __init__(self, folder, patch_size=96, scale=4):
        self.files = sorted(Path(folder).glob("*.png"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"nenhum .png encontrado em '{folder}'. verifique o caminho.")
        self.patch_size = patch_size
        self.scale = scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size

        # guard: imagens menores que patch_size quebrariam o randint
        if w < self.patch_size or h < self.patch_size:
            img = img.resize(
                (max(w, self.patch_size), max(h, self.patch_size)),
                Image.BICUBIC,
            )
            w, h = img.size

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        hr = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # flip horizontal + rotacoes de 90 graus
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
    # dataset de avaliacao simples: usa a imagem inteira, gera o lr via bicubic
    # usado quando so tem imagens hr soltas (sem pares lr oficiais), tipo div2k_valid_hr

    def __init__(self, folder, scale=4):
        self.files = sorted(Path(folder).glob("*.png"))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"nenhum .png encontrado em '{folder}'. verifique o caminho "
                f"(ex: pasta extraida corretamente)."
            )
        self.scale = scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size

        # garante que w,h sejam multiplos do scale (evita erro de dimensao)
        w = w - (w % self.scale)
        h = h - (h % self.scale)
        hr = img.crop((0, 0, w, h))

        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)

        return TF.to_tensor(lr), TF.to_tensor(hr)


class SRDatasetAugmented(Dataset):
    # dataset de treino com augmentacao robusta
    # tecnicas: flip horizontal, rotacao arbitraria, zoom aleatorio,
    # brightness/contrast, gaussian blur, color jittering

    def __init__(self, folder, patch_size=96, scale=4):
        self.files = sorted(Path(folder).glob("*.png"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"nenhum .png encontrado em '{folder}'.")
        self.patch_size = patch_size
        self.scale = scale
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

    def __len__(self):
        return len(self.files)

    def _apply_augmentations(self, img):
        # aplica a lista de tecnicas de augmentacao

        # flip horizontal (50%)
        if random.random() < 0.5:
            img = TF.hflip(img)

        # rotacao arbitraria (-45 a +45)
        if random.random() < 0.7:
            angle = random.uniform(-45, 45)
            img = TF.rotate(img, angle, expand=False)

        # zoom/crop aleatorio
        if random.random() < 0.6:
            scale_factor = random.uniform(0.85, 1.15)
            new_size = int(self.patch_size * scale_factor)
            img = img.resize((new_size, new_size), Image.BICUBIC)

        # brightness/contrast
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.85, 1.15)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)

        if random.random() < 0.5:
            contrast_factor = random.uniform(0.85, 1.15)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)

        # gaussian blur (simula degradacao real)
        if random.random() < 0.4:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # color jittering
        if random.random() < 0.5:
            img = self.color_jitter(img)

        return img

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size

        if w < self.patch_size or h < self.patch_size:
            img = img.resize(
                (max(w, self.patch_size), max(h, self.patch_size)),
                Image.BICUBIC,
            )
            w, h = img.size

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # aplica augmentacoes
        hr = self._apply_augmentations(hr)

        # redimensiona pro patch_size (pode ter crescido/diminuido)
        hr = hr.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        lr = hr.resize(
            (self.patch_size // self.scale, self.patch_size // self.scale),
            Image.BICUBIC,
        )

        return TF.to_tensor(lr), TF.to_tensor(hr)


class SRDatasetMultiScale(Dataset):
    # dataset que treina em multiplas escalas (2x, 3x, 4x)
    # a cada getitem, escolhe uma escala aleatoria diferente
    # retorna so (lr, hr), a escala usada nao e retornada

    def __init__(self, folder, patch_size=96, scales=None):
        self.files = sorted(Path(folder).glob("*.png"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"nenhum .png encontrado em '{folder}'.")
        self.patch_size = patch_size
        self.scales = scales if scales is not None else [2, 3, 4]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size

        if w < self.patch_size or h < self.patch_size:
            img = img.resize(
                (max(w, self.patch_size), max(h, self.patch_size)),
                Image.BICUBIC,
            )
            w, h = img.size

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        hr = img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # tratamento basico
        if random.random() < 0.5:
            hr = TF.hflip(hr)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            hr = hr.rotate(angle)

        # escolhe escala aleatoria e gera lr
        scale = random.choice(self.scales)
        lr = hr.resize(
            (self.patch_size // scale, self.patch_size // scale),
            Image.BICUBIC,
        )

        return TF.to_tensor(lr), TF.to_tensor(hr)


class SRBenchmarkDataset(Dataset):
    # carrega pares hr/lr prontos de um benchmark (set5, set14, bsd100, urban100)
    #
    # espera uma estrutura como:
    #   <folder>/img_001_SRF_4_HR.png
    #   <folder>/img_001_SRF_4_LR.png
    #   <folder>/img_002_SRF_4_HR.png
    #   <folder>/img_002_SRF_4_LR.png
    #
    # diferente do srevaldataset (que gera o lr na hora via bicubic), aqui
    # usa o lr oficial do benchmark, o que deixa os resultados comparaveis
    # com os papers de referencia (dong, lim, wang)

    def __init__(self, folder, scale=4):
        folder = Path(folder)
        hr_files = sorted(folder.glob("*_HR.png"))

        if len(hr_files) == 0:
            raise FileNotFoundError(
                f"nenhum arquivo '*_HR.png' encontrado em '{folder}'. "
                f"confirme o caminho (ex: 'datasets/Set5/image_SRF_{scale}')."
            )

        self.pairs = []
        for hr_path in hr_files:
            # opera so no nome do arquivo, nao no caminho completo, pra
            # evitar substituicao indesejada caso o caminho contenha "_HR.png"
            lr_name = hr_path.name.replace("_HR.png", "_LR.png")
            lr_path = hr_path.with_name(lr_name)

            if not lr_path.exists():
                raise FileNotFoundError(
                    f"par lr nao encontrado para '{hr_path.name}' "
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

        # valida se a razao real hr/lr bate com o scale informado
        # so checa uma vez (primeiro item) pra nao repetir aviso
        if not self._scale_checked:
            real_ratio_w = hw / lw
            real_ratio_h = hh / lh
            if abs(real_ratio_w - self.scale) > 0.5 or abs(real_ratio_h - self.scale) > 0.5:
                print(
                    f"[aviso] scale={self.scale} informado, mas a razao real "
                    f"hr/lr e ~{real_ratio_w:.2f}x. confirme se a pasta "
                    f"'{lr_path.parent}' corresponde ao fator de escala correto."
                )
            self._scale_checked = True

        expected_hw, expected_hh = lw * self.scale, lh * self.scale
        if (hw, hh) != (expected_hw, expected_hh):
            hr = hr.crop((0, 0, expected_hw, expected_hh))

        return TF.to_tensor(lr), TF.to_tensor(hr)