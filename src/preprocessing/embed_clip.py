# embed_clip.py
import os
from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def list_images(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)

@torch.no_grad()
def compute_clip_embeddings(
    image_paths: List[Path],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    batch_size: int = 128,
    device: str = "cuda",
    out_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, List[str]]:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    all_embeds = []
    all_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP embed"):
        batch_paths = image_paths[i:i+batch_size]
        imgs = []
        keep = []
        for p in batch_paths:
            try:
                im = Image.open(p).convert("RGB")
                imgs.append(preprocess(im))
                keep.append(str(p))
            except Exception:
                # skip unreadable images
                continue

        if not imgs:
            continue

        x = torch.stack(imgs, dim=0).to(device)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize to unit length
        feats = feats.to(out_dtype).cpu()

        all_embeds.append(feats)
        all_paths.extend(keep)

    embeds = torch.cat(all_embeds, dim=0)
    return embeds, all_paths

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--out", type=str, default="embeddings.pt")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.image_root)
    paths = list_images(root)
    print("found images:", len(paths))

    embeds, kept_paths = compute_clip_embeddings(
        paths,
        batch_size=args.batch_size,
        device=args.device,
    )

    payload = {
        "embeds": embeds,          # (N, D)
        "paths": kept_paths,       # len N
        "model": "ViT-B-32/laion2b_s34b_b79k",
    }
    torch.save(payload, args.out)
    print("saved:", args.out, "embeds:", tuple(embeds.shape))

if __name__ == "__main__":
    main()
