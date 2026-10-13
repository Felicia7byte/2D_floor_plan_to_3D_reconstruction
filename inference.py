from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from model import FloorPlanUNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_SIZE = 256

CLASS_NAMES = {
    0: "background",
    1: "floor",
    2: "wall",
    3: "door",
    4: "window",
}


def load_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint_path}"
        )

    # Checkpoint ini berasal dari training milik sendiri.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Kompatibilitas dengan versi PyTorch lama.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    # Training kamu menyimpan model_state_dict.
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Key 'model_state_dict' tidak ditemukan. "
            "Periksa format checkpoint."
        )

    # Baca konfigurasi dari checkpoint.
    config = checkpoint.get("config", {})

    base_channels = config.get("base_channels", 32)
    num_classes = config.get("num_classes", 5)
    image_size = config.get("image_size", 256)

    if num_classes != 5:
        raise ValueError(
            f"Checkpoint memiliki {num_classes} kelas, "
            "sedangkan aplikasi mengharapkan 5 kelas."
        )

    if image_size != INPUT_SIZE:
        raise ValueError(
            f"Ukuran training {image_size} tidak sama "
            f"dengan INPUT_SIZE={INPUT_SIZE}."
        )

    model = FloorPlanUNet(
        in_channels=3,
        num_classes=num_classes,
        base=base_channels,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print(f"Checkpoint berhasil dimuat: {checkpoint_path}")
    print(f"Device: {DEVICE}")
    print(f"Base channels: {base_channels}")
    print(f"Jumlah kelas: {num_classes}")

    return model


def predict_mask(model, image):
    original_size = image.size

    # Samakan ukuran dan preprocessing dengan training.
    image = image.convert("RGB")
    image = TF.resize(
        image,
        [INPUT_SIZE, INPUT_SIZE],
        interpolation=InterpolationMode.BILINEAR,
    )

    tensor = TF.to_tensor(image)

    # HARUS sama dengan normalisasi saat training:
    # mean=(0.5, 0.5, 0.5)
    # std=(0.5, 0.5, 0.5)
    tensor = TF.normalize(
        tensor,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        prediction = torch.argmax(logits, dim=1)[0]

    mask = prediction.cpu().numpy().astype(np.uint8)

    # Kembalikan ukuran mask ke ukuran input asli.
    # Nearest-neighbor menjaga ID kelas tetap utuh.
    mask_image = Image.fromarray(mask)
    mask_image = mask_image.resize(
        original_size,
        Image.Resampling.NEAREST,
    )

    return np.asarray(mask_image, dtype=np.uint8)