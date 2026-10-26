"""CLIP similarity between a text description and street view images, plus privacy-blur detection.

Blur detection is a texture heuristic rather than CLIP: CLIP zero-shot could not tell blurred from clear views.
"""

import numpy as np
import torch
from PIL import Image, ImageFilter

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

PROMPT_TEMPLATES = [
    "a google street view photo of {}",
    "a photo of {}",
    "{}",
]

class Scorer:
    def __init__(self, device=None):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    @torch.no_grad()
    def _encode_text(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(self, images, batch_size=32):
        out = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.preprocess(im) for im in images[i : i + batch_size]]).to(self.device)
            feats = self.model.encode_image(batch).float()
            out.append(feats / feats.norm(dim=-1, keepdim=True))
        return torch.cat(out) if out else torch.empty(0)

    def description_embedding(self, description):
        feats = self._encode_text([t.format(description) for t in PROMPT_TEMPLATES])
        mean = feats.mean(dim=0)
        return mean / mean.norm()

    def similarity(self, image_feats, text_feat):
        return (image_feats @ text_feat).cpu().numpy()


def blur_map(image, tile=32):
    """Per-tile flags for 'smooth but not flat' regions: the signature of Google's privacy blur.

    Real surfaces keep fine detail (texture, noise) wherever they have coarse structure. A heavily
    blurred region still has coarse shading from what was underneath, but almost no fine detail.
    Flat areas (sky, plain walls) have neither and are ignored.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    coarse = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)
    fine = gray - np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)
    gy, gx = np.gradient(coarse)
    coarse_grad = np.hypot(gx, gy)

    rows, cols = gray.shape[0] // tile, gray.shape[1] // tile
    flags = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            sl = (slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile))
            structure = coarse_grad[sl].mean()
            detail = np.abs(fine[sl]).mean()
            flags[r, c] = structure > 0.6 and detail < 0.45 and detail / structure < 0.35
    return flags


def largest_blob(flags):
    """Size of the biggest 4-connected group of flagged tiles."""
    seen = np.zeros_like(flags)
    best = 0
    rows, cols = flags.shape
    for r in range(rows):
        for c in range(cols):
            if not flags[r, c] or seen[r, c]:
                continue
            stack, size = [(r, c)], 0
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                size += 1
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < rows and 0 <= nx < cols and flags[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            best = max(best, size)
    return best


def blur_fraction(image):
    """Fraction of a level (pitch 0) view, from just above the horizon down, covered by the largest blurred blob.

    Skipping the upper part keeps smooth clouds and sky gradients from looking like a privacy blur.
    """
    w, h = image.size
    region = image.crop((0, int(h * 0.38), w, h))
    flags = blur_map(region)
    return largest_blob(flags) / max(flags.size, 1)
