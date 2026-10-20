# 输入为npy的纯净版 (集成对齐与增强可视化)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib

matplotlib.use("TkAgg")

# =====================================================
# 基本设置
# =====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 512
wavelength = 532e-9
pixel_size = 8e-6
z = 45e-3
noise_std_init = 0.01
num_steps = 10001
lr = 1e-3


# =====================================================
# 指标计算
# =====================================================
# def calculate_metrics(rec, gt):
#     offset = np.mean(rec) - np.mean(gt)
#     rec_aligned = rec - offset
#     mse = np.mean((rec_aligned - gt) ** 2)
#     psnr = peak_signal_noise_ratio(gt, rec_aligned, data_range=2 * np.pi)
#     ssim = structural_similarity(gt, rec_aligned, data_range=2 * np.pi)
#     return mse, psnr, ssim, rec_aligned
def calculate_metrics(rec, gt):
    """
    增强版对齐函数：消除常数偏移和线性倾斜
    """
    n = rec.shape[0]
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    # 1. 计算原始差值
    diff = rec - gt

    # 2. 最小二乘法拟合平面 Ax + By + C
    A_mat = np.stack([X.ravel(), Y.ravel(), np.ones_like(X).ravel()], axis=-1)
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, diff.ravel(), rcond=None)

    # 3. 生成拟合的倾斜平面并减去
    tilt_plane = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    rec_aligned = rec - tilt_plane

    # 4. 计算指标
    mse = np.mean((rec_aligned - gt) ** 2)
    psnr = peak_signal_noise_ratio(gt, rec_aligned, data_range=1 * np.pi)
    ssim = structural_similarity(gt, rec_aligned, data_range=1 * np.pi)

    return mse, psnr, ssim, rec_aligned


# =====================================================
# Fresnel 传播
# =====================================================


def fresnel_kernel(Z):
    fx = torch.fft.fftfreq(N, pixel_size).to(device)
    FX, FY = torch.meshgrid(fx, fx, indexing='ij')
    return torch.exp(-1j * np.pi * wavelength * Z * (FX ** 2 + FY ** 2))


H_fwd = fresnel_kernel(z)
H_bwd = torch.conj(H_fwd)


def fresnel(U_in, H):
    return torch.fft.ifft2(torch.fft.fft2(U_in) * H)


# =====================================================
# DIP-UNet
# =====================================================


class DIP_UNet(nn.Module):
    def __init__(self, base_c=32):
        super().__init__()

        def CBR(j, o):
            return nn.Sequential(
                nn.Conv2d(j, o, 3, padding=1),
                # nn.InstanceNorm2d(o, affine=True),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.enc1 = CBR(1, base_c)
        self.enc2 = CBR(base_c, base_c * 2)
        self.enc3 = CBR(base_c * 2, base_c * 4)
        # self.pool = nn.AvgPool2d(2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = CBR(base_c * 4, base_c * 8)
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, 2, 2)
        self.dec3 = CBR(base_c * 8, base_c * 4)
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, 2)
        self.dec2 = CBR(base_c * 4, base_c * 2)
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, 2)
        self.dec1 = CBR(base_c * 2, base_c)
        self.out_conv = nn.Conv2d(base_c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out_conv(d1)) * 2 * np.pi


# =====================================================
# 主程序
# =====================================================


if __name__ == '__main__':

    try:
        phase_gt_np = np.load('./database/output6.npy')
        if phase_gt_np.shape != (N, N):
            phase_gt_np = cv2.resize(phase_gt_np, (N, N))
    except FileNotFoundError:
        raise FileNotFoundError("未找到.npy文件，请确保生成脚本已运行。")

    phase_gt = torch.tensor(phase_gt_np, device=device)

    # Forward：生成衍射强度
    with torch.no_grad():
        U0 = torch.exp(1j * phase_gt)
        Ud = fresnel(U0, H_fwd)
        I_meas = torch.abs(Ud) ** 2

    I_meas_input = I_meas.unsqueeze(0).unsqueeze(0)

    # PI-UNet 优化
    net = DIP_UNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    phys_losses = []

    print('Start PI-UNet optimization...')
    for step in range(num_steps):
        if step < (num_steps - 1) * 0.95:
            curr_std = noise_std_init
        else:
            curr_std = 0
        noise = torch.randn_like(I_meas_input) * curr_std
        I_noisy = torch.clamp(I_meas_input + noise, min=0.0)
        phase_pred = net(I_noisy)
        U0_pred = torch.exp(1j * phase_pred.squeeze())
        Ud_pred = fresnel(U0_pred, H_fwd)
        I_pred = torch.abs(Ud_pred) ** 2

        loss = F.mse_loss(I_pred, I_meas)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        phys_losses.append(loss.item())
        if step % 500 == 0:
            print(f'[PI-UNet] {step:05d} | MSE = {loss.item():.4e}')

    # phase_phys = phase_pred.squeeze().detach().cpu().numpy()
    net.eval()
    with torch.no_grad():
        phase_phys = net(I_meas_input).squeeze().cpu().numpy()

    # -------- 多平面 z 设置（论文对齐）--------
    num_planes = 3
    z_list = torch.linspace(
        z * 0.8, z * 1.2, num_planes, device=device
    )

    H_fwd_list = [fresnel_kernel(zi) for zi in z_list]
    H_bwd_list = [torch.conj(H) for H in H_fwd_list]

    print('Start GS Algorithm...')

    # -------- 多平面幅度测量 --------
    amp_meas_list = []

    with torch.no_grad():
        for zi, Hzi in zip(z_list, H_fwd_list):
            Ud_zi = fresnel(U0, Hzi)
            I_zi = torch.abs(Ud_zi) ** 2
            amp_meas_list.append(torch.sqrt(I_zi + 1e-12))

    # GS 初始物面
    U0_gs = torch.exp(1j * (torch.rand((N, N), device=device) * 0.1))
    # GS 迭代
    for i in range(1201):

        U = U0_gs

        # -------- 依次经过多个 z 平面 --------
        for Hf, Hb, amp_meas in zip(H_fwd_list, H_bwd_list, amp_meas_list):
            # 前向传播
            Ud = torch.fft.ifft2(torch.fft.fft2(U) * Hf)

            # 幅度替换（全场）
            Ud = amp_meas * torch.exp(1j * torch.angle(Ud))

            # 反向传播
            U = torch.fft.ifft2(torch.fft.fft2(Ud) * Hb)

        # -------- 物面约束（相位物体）--------
        U0_gs = torch.exp(1j * torch.angle(U))
        # actual_phase = torch.remainder(torch.angle(U), 2 * np.pi)
        # U0_gs = torch.exp(1j * actual_phase)

        if i % 400 == 0:
            torch.cuda.empty_cache()
            phase_rec = torch.angle(U0_gs)
            mse_tmp = torch.mean((phase_rec - phase_gt) ** 2).item()
            print(f'[GS-multiZ] {i:05d} | MSE = {mse_tmp:.4e}')

    # -------------------------------
    # 输出相位
    # -------------------------------
    phase_gs = torch.angle(U0_gs).cpu().numpy()

    # =================================================
    # 指标计算 (带相位对齐)
    # =================================================
    mse_p, psnr_p, ssim_p, rec_p = calculate_metrics(phase_phys, phase_gt_np)
    mse_g, psnr_g, ssim_g, rec_g = calculate_metrics(phase_gs, phase_gt_np)

    print('\n========== Metrics (Aligned) ==========')
    print(f'PI-UNet   | PSNR: {psnr_p:.2f}, SSIM: {ssim_p:.3f}')
    print(f'GS        | PSNR: {psnr_g:.2f}, SSIM: {ssim_g:.3f}')

    # =================================================
    # 可视化
    # =================================================
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    # plt.imshow(phase_gt_np, cmap='gray')
    # plt.title('GT Phase')
    gt_phase = plt.imshow(phase_gt_np / (2 * np.pi), cmap='gray', vmin=0, vmax=0.5)
    cbar_gt = plt.colorbar(gt_phase, fraction=0.046, pad=0.04)
    cbar_gt.set_label(r'$(2\pi \ rad)$')
    plt.axis("off")
    plt.title('GT Phase')

    plt.subplot(2, 3, 2)
    plt.imshow(I_meas.cpu().numpy(), cmap='magma')
    plt.axis("off")
    plt.title('Measured Intensity')

    plt.subplot(2, 3, 3)
    plt.plot(phys_losses)
    plt.yscale('log')
    plt.title('PI-UNet Loss (Intensity MSE)')

    plt.subplot(2, 3, 4)
    # plt.imshow(rec_p, cmap='gray')
    # plt.title(f'PI-UNet (PSNR: {psnr_p:.2f}  SSIM: {ssim_p:.3f})')
    phy = plt.imshow(rec_p / (2 * np.pi), cmap='gray', vmin=0, vmax=0.5)
    cbar_phy = plt.colorbar(phy, fraction=0.046, pad=0.04)
    cbar_phy.set_label(r'$(2\pi \ rad)$')
    plt.axis("off")
    plt.title(f'PI-UNet (PSNR: {psnr_p:.2f}  SSIM: {ssim_p:.3f})')

    plt.subplot(2, 3, 5)
    # plt.imshow(rec_g, cmap='gray')
    # plt.title(f'GS (PSNR: {psnr_g:.2f}  SSIM: {ssim_g:.3f})')
    gs = plt.imshow(rec_g / (2 * np.pi), cmap='gray', vmin=0, vmax=0.5)
    cbar_gs = plt.colorbar(gs, fraction=0.046, pad=0.04)
    cbar_gs.set_label(r'$(2\pi \ rad)$')
    plt.axis("off")
    plt.title(f'GS (PSNR: {psnr_g:.2f}  SSIM: {ssim_g:.3f})')

    plt.subplot(2, 3, 6)
    error = rec_p - phase_gt_np
    # 这样 0 是白色，红色是偏大，蓝色是偏小
    v_limit = max(abs(error.min()), abs(error.max()))
    plt.imshow(error, cmap='seismic', vmin=-v_limit, vmax=v_limit)
    plt.colorbar()
    plt.axis("off")
    plt.title('Error Map')

    plt.tight_layout()
    plt.show()
    torch.cuda.empty_cache()
