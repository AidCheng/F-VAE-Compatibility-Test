from math import log10, sqrt
import cv2
import numpy as np

from SSIM_PIL import compare_ssim
from PIL import Image

import lpips
import torch
import torchvision.transforms as transforms

def PSNR(ori_path, com_path):
    original = cv2.imread(ori_path)
    compressed = cv2.imread(com_path, 1)

    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(ori_path, com_path):
    original = Image.open(ori_path)
    compressed = Image.open(com_path)
    value = compare_ssim(original, compressed)
    return value

def LPIPS(ori_path, com_path):
    loss_fn = lpips.LPIPS(net='alex')  # 可以换成 'vgg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = loss_fn.to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # LPIPS 输入需统一尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image1 = transform(Image.open(ori_path).convert("RGB")).unsqueeze(0).to(device)
    image2 = transform(Image.open(com_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        dist = loss_fn(image1, image2).item()
    return dist

def main():
    original = "input.png"
    compressed = "output.png"

    v_psnr = PSNR(original, compressed)
    v_ssim = SSIM(original, compressed)
    v_lpips = LPIPS(original, compressed)

    print(f"PSNR value is {v_psnr:.2f} dB")
    print(f"SSIM index is {v_ssim:.4f}")
    print(f"LPIPS distance is {v_lpips:.4f}")

if __name__ == "__main__":
    main()