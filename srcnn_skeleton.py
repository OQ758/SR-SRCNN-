# srcnn_skeleton.py - SRCNN Super-Resolution Feature Extraction Module
# This module can be used as a preprocessing step before YOLOv5 detection

import os
import random
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np


# ============================
# 1. 数据集模块
# ============================
class SRDataset(Dataset):
    """HR -> LR -> 上采样后的网络输入"""

    def __init__(self, root, scale=2, patch_size=48):
        self.root = root
        self.files = glob(os.path.join(root, "*.png")) + glob(os.path.join(root, "*.jpg"))
        self.scale = scale
        self.patch_size = patch_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def _random_crop(self, img):
        """随机裁剪 patch"""
        w, h = img.size
        pw, ph = self.patch_size, self.patch_size
        if w < pw or h < ph:
            img = transforms.Resize((max(ph, h), max(pw, w)))(img)
            w, h = img.size
        left = random.randint(0, w - pw)
        top = random.randint(0, h - ph)
        return img.crop((left, top, left + pw, top + ph))

    def __getitem__(self, idx):
        """打开 HR 图像 -> 随机裁剪 -> 下采样生成 LR -> 上采样回 HR 大小 -> 转 tensor 返回"""
        img_path = self.files[idx]
        hr = Image.open(img_path).convert("RGB")

        # 随机裁剪
        hr = self._random_crop(hr)

        # 获取原始尺寸
        w, h = hr.size

        # 下采样生成 LR
        lr_size = (w // self.scale, h // self.scale)
        lr = hr.resize(lr_size, Image.BICUBIC)

        # 上采样回 HR 大小
        sr = lr.resize((w, h), Image.BICUBIC)

        # 转 tensor
        hr_tensor = self.to_tensor(hr)  # [C, H, W], range [0, 1]
        sr_tensor = self.to_tensor(sr)  # [C, H, W], range [0, 1]

        return sr_tensor, hr_tensor


# ============================
# 2. 模型模块
# ============================
class SRCNN(nn.Module):
    """SRCNN 网络 - Super-Resolution Convolutional Neural Network
    结构: Conv1(9x9) -> ReLU -> Conv2(1x1) -> ReLU -> Conv3(5x5) -> 输出
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # 定义三层卷积
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=1 // 2)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """实现 forward
        conv1 -> relu -> conv2 -> relu -> conv3 -> 输出 clamp
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return torch.clamp(x, 0.0, 1.0)


# ============================
# 3. 工具函数模块
# ============================
def calc_psnr(sr, hr, shave_border=0):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    if shave_border > 0:
        sr = sr[..., shave_border:-shave_border, shave_border:-shave_border]
        hr = hr[..., shave_border:-shave_border, shave_border:-shave_border]
    
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = 1.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


# ============================
# 4. 训练循环模块
# ============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一轮"""
    model.train()
    total_loss = 0.0
    
    for sr, hr in tqdm(loader, desc="Training"):
        sr, hr = sr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        output = model(sr)
        loss = criterion(output, hr)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, device):
    """验证并计算平均 PSNR"""
    model.eval()
    total_psnr = 0.0
    
    with torch.no_grad():
        for sr, hr in tqdm(loader, desc="Validating"):
            sr, hr = sr.to(device), hr.to(device)
            output = model(sr)
            psnr = calc_psnr(output, hr)
            total_psnr += psnr
    
    return total_psnr / len(loader)


# ============================
# 5. 主函数
# ============================
def main():
    # 数据集路径 - 使用 dataset/images 下的 train 和 val 子文件夹
    train_dir = "./dataset/images/train"
    val_dir = "./dataset/images/val"
    scale = 2
    batch_size = 8
    epochs = 30
    lr = 1e-4
    patch_size = 48
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")

    # 创建 dataset 和 dataloader
    train_dataset = SRDataset(train_dir, scale=scale, patch_size=patch_size)
    val_dataset = SRDataset(val_dir, scale=scale, patch_size=patch_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 初始化 SRCNN 模型
    model = SRCNN(in_channels=3).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环 + 验证 + 保存最优模型
    best_psnr = 0.0
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_psnr = validate(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val PSNR: {val_psnr:.2f} dB")
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "best_srcnn.pth")
            print(f"  -> Saved best model with PSNR: {best_psnr:.2f} dB")

    print(f"Training completed! Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
