from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# REPLACE WITH REAL IMAGES ONCE DONE, simple sampling gan

class ClipCondImageDataset(Dataset):
    def __init__(self, payload_path: str, image_size: int = 64):
        payload = torch.load(payload_path, map_location="cpu")
        self.embeds = payload["embeds"].float()
        self.paths: List[str] = payload["paths"]

        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        x = Image.open(p).convert("RGB")
        x = self.tf(x)
        c = self.embeds[idx]
        c = c / (c.norm() + 1e-8)
        return x, c

# simple gan
class CondProjector(nn.Module):
    def __init__(self, cond_dim=512, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out),
        )
    def forward(self, c):
        return self.net(c)

class Generator(nn.Module):
    def __init__(self, z_dim=128, cond_dim=512, c_proj=128, img_ch=3, base=256):
        super().__init__()
        self.cproj = CondProjector(cond_dim, out=c_proj)
        self.fc = nn.Linear(z_dim + c_proj, base * 4 * 4)

        def up(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            )

        self.up1 = up(base, base//2)
        self.up2 = up(base//2, base//4)
        self.up3 = up(base//4, base//8)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base//8, base//16, 4, 2, 1),
            nn.BatchNorm2d(base//16),
            nn.SiLU(),
            nn.Conv2d(base//16, img_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, c):
        cp = self.cproj(c)
        h = torch.cat([z, cp], dim=1)
        h = self.fc(h).view(h.size(0), -1, 4, 4)
        h = self.up1(h); h = self.up2(h); h = self.up3(h); x = self.up4(h)

        return x



class Discriminator(nn.Module):
    def __init__(self, cond_dim=512, c_proj=128, img_ch=3, base=64):
        super().__init__()
        self.cproj = CondProjector(cond_dim, out=c_proj)

        def down(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.d1 = down(img_ch, base)
        self.d2 = down(base, base*2)
        self.d3 = down(base*2, base*4)
        self.d4 = down(base*4, base*8)

        self.fc = nn.Linear(base*8*4*8 + c_proj, 1)

    def forward(self, x, c):
        h = self.d1(x); h = self.d2(h); h = self.d3(h); h = self.d4(h)
        h = h.flatten(1)
        cp = self.cproj(c)
        h = torch.cat([h, cp], dim=1)
        return self.fc(h).squeeze(1)

# -------------------------
# Train
# -------------------------
@dataclass
class TrainCfg:
    payload_path: str = "emb.pt"
    image_size: int = 64
    z_dim: int = 128
    batch_size: int = 64
    lr: float = 2e-4
    steps: int = 20_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "runs/cgan"

def bce_logits(logits, target):
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, float(target)))

@torch.no_grad()
def save_grid(gen, dataset, cfg: TrainCfg, step: int):
    from torchvision.utils import make_grid, save_image
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # sample few random cond from real images
    idx = torch.randint(0, len(dataset), (16,))
    c = torch.stack([dataset[i][1] for i in idx]).to(cfg.device)
    z = torch.randn(16, cfg.z_dim, device=cfg.device)

    x = gen(z, c)
    grid = make_grid((x + 1) / 2, nrow=4)
    save_image(grid, f"{cfg.out_dir}/sample_{step:07d}.png")

def main():
    cfg = TrainCfg()
    ds = ClipCondImageDataset(cfg.payload_path, image_size=cfg.image_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)

    G = Generator(z_dim=cfg.z_dim).to(cfg.device)
    D = Discriminator().to(cfg.device)

    optG = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    step = 0
    it = iter(dl)

    pbar = tqdm(total=cfg.steps)
    while step < cfg.steps:
        try:
            real, c = next(it)
        except StopIteration:
            it = iter(dl)
            real, c = next(it)

        real = real.to(cfg.device)
        c = c.to(cfg.device)
        b = real.size(0)

        z = torch.randn(b, cfg.z_dim, device=cfg.device)
        fake = G(z, c).detach()

        d_real = D(real, c)
        d_fake = D(fake, c)

        lossD = bce_logits(d_real, 1.0) + bce_logits(d_fake, 0.0)

        optD.zero_grad(set_to_none=True)
        lossD.backward()
        optD.step()

        z = torch.randn(b, cfg.z_dim, device=cfg.device)
        fake = G(z, c)
        d_fake = D(fake, c)
        lossG = bce_logits(d_fake, 1.0)

        optG.zero_grad(set_to_none=True)
        lossG.backward()
        optG.step()

        if step % 500 == 0:
            save_grid(G, ds, cfg, step)

        pbar.update(1)
        pbar.set_description(f"lossD={lossD.item():.3f} lossG={lossG.item():.3f}")
        step += 1

    pbar.close()

if __name__ == "__main__":
    main()
