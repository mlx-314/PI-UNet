# PI-UNet+高斯噪声
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
# 1. 基本参数设置
# =====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 512
wavelength = 532e-9
pixel_size = 8e-6
z = 45e-3
num_steps = 10001
lr = 1e-3
noise_std_init = 0.01
test_noise_std = 0.05  # 测试用的噪声水平（标准差）
# TV正则化权重（如果觉得图还是太花，可以调大）
tv_weight = 8e-5  # 原5e-6，5e-5，8e-5
phase_range = 2 * np.pi
# 输入的相位图的相位范围
phase_min, phase_max = 0, 0.5

# =====================================================
# 2. 物理模型
# =====================================================
# def fresnel_kernel(z, size):
#     fx = torch.fft.fftfreq(size, pixel_size).to(device)
#     FX, FY = torch.meshgrid(fx, fx, indexing='ij')
#     return torch.exp(-1j * np.pi * wavelength * z * (FX ** 2 + FY ** 2))


def fresnel_kernel(Z, size):
    f = torch.fft.fftfreq(size, pixel_size).to(device)
    FX, FY = torch.meshgrid(f, f, indexing='ij')
    F_rho2 = FX ** 2 + FY ** 2
    H = torch.exp(-1j * np.pi * wavelength * Z * F_rho2)

    # 临界频率处理 (保持原有的带限滤波逻辑，但尺寸改为 size)
    f_nyquist = 1 / (2 * pixel_size)
    L = size * pixel_size
    f_geometric = L / (2 * wavelength * Z + 1e-12)
    f_limit = min(f_nyquist, f_geometric)
    if f_nyquist < f_geometric:
        return H
    else:
        F_rho = torch.sqrt(F_rho2)
        mask = torch.sigmoid((f_limit - F_rho) / (0.02 * f_limit))
        return H * mask.float()


def fresnel_propagate(U_in, H):
    U_fft = torch.fft.fft2(U_in)
    return torch.fft.ifft2(U_fft * H)


# TV Loss 计算函数
def calculate_tv_loss(img):
    """计算全变分，抑制重建图中的高频波纹"""
    # img 维度为 [1, 1, H, W]
    tv_h = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


# =====================================================
# 3. 网络架构
# =====================================================
class DIP_UNet(nn.Module):
    def __init__(self, base_c=32):
        super().__init__()

        def CBR(j, o):
            return nn.Sequential(
                nn.Conv2d(j, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.enc1 = CBR(1, base_c)
        self.enc2 = CBR(base_c, base_c * 2)
        self.enc3 = CBR(base_c * 2, base_c * 4)
        self.pool = nn.AvgPool2d(2)
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
        # return (torch.sin(self.out_conv(d1)) + 1.0) * np.pi


# =====================================================
# 4. 指标计算与相位对齐
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
    # 构造设计矩阵 [X, Y, 1]
    A_mat = np.stack([X.ravel(), Y.ravel(), np.ones_like(X).ravel()], axis=-1)
    # 求解系数 [A, B, C]
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, diff.ravel(), rcond=None)

    # 3. 生成拟合的倾斜平面并减去
    tilt_plane = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    rec_aligned = rec - tilt_plane

    # 4. 计算指标
    mse = np.mean((rec_aligned - gt) ** 2)
    psnr = peak_signal_noise_ratio(gt, rec_aligned, data_range=1 * np.pi)
    ssim = structural_similarity(gt, rec_aligned, data_range=1 * np.pi)

    return mse, psnr, ssim, rec_aligned
# def calculate_metrics(rec, gt):
#     """
#     二阶拟合对齐函数：消除常数、线性倾斜以及二阶曲面（球差/场曲）偏差
#     """
#     N = rec.shape[0]
#     # 建立归一化坐标系 [-1, 1]
#     x = np.linspace(-1, 1, N)
#     y = np.linspace(-1, 1, N)
#     X, Y = np.meshgrid(x, y)
#
#     # 1. 计算原始差值
#     diff = rec - gt
#
#     # 2. 构造二阶设计矩阵: [1, x, y, x^2, y^2, xy]
#     X_flat = X.ravel()
#     Y_flat = Y.ravel()
#     # 这一步包含了常数项、线性项和二阶项
#     A_mat = np.stack([
#         np.ones_like(X_flat),  # 常数项
#         X_flat,  # 线性 X
#         Y_flat,  # 线性 Y
#         X_flat ** 2,  # 二阶 X^2
#         Y_flat ** 2,  # 二阶 Y^2
#         X_flat * Y_flat  # 交叉项 XY
#     ], axis=-1)
#
#     # 3. 最小二乘求解系数
#     coeffs, _, _, _ = np.linalg.lstsq(A_mat, diff.ravel(), rcond=None)
#
#     # 4. 生成拟合的误差曲面并减去
#     # fit_surface = C + Ax + By + Dx^2 + Ey^2 + Fxy
#     fit_surface = (coeffs[0] + coeffs[1] * X + coeffs[2] * Y +
#                    coeffs[3] * X ** 2 + coeffs[4] * Y ** 2 + coeffs[5] * X * Y)
#
#     rec_aligned = rec - fit_surface
#
#     # 5. 计算指标
#     mse = np.mean((rec_aligned - gt) ** 2)
#     psnr = peak_signal_noise_ratio(gt, rec_aligned, data_range=2 * np.pi)
#     ssim = structural_similarity(gt, rec_aligned, data_range=2 * np.pi)
#
#     return mse, psnr, ssim, rec_aligned


# =====================================================
# 5. 主程序
# =====================================================

if __name__ == '__main__':
    try:
        phase_gt_np = np.load('./database/output7.npy')
        # 确保尺寸匹配
        if phase_gt_np.shape != (N, N):
            phase_gt_np = cv2.resize(phase_gt_np, (N, N))
    except FileNotFoundError:
        raise FileNotFoundError("未找到 output.npy 文件，请先运行生成脚本。")

    phase_gt = torch.tensor(phase_gt_np, device=device)
    # ------------------------------------------

    H_fwd = fresnel_kernel(z, N)

    with torch.no_grad():
        U0 = torch.exp(1j * phase_gt)
        Ud = fresnel_propagate(U0, H_fwd)
        I_meas_clean = torch.abs(Ud) ** 2

        # --- 添加噪声 ---
        noise = test_noise_std * torch.randn_like(I_meas_clean)
        I_meas = torch.clamp(I_meas_clean + noise, min=0.0)  # 实际观测到的带噪图

    I_meas_input = I_meas.unsqueeze(0).unsqueeze(0)

    net = DIP_UNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=8e-6)  # 原8e-6，1e-5

    phys_losses = []

    print(f'Start Optimization (z={z * 1000:.1f}mm)...')
    for step in range(num_steps):
        decay_limit = 0.95  # 在 95% 的进度时达到噪声最低点并关闭
        if step < (num_steps-1) * decay_limit:
            # 在前 95% 阶段从初始值衰减到万分之一
            curr_std = noise_std_init * (1e-4 ** (step / (num_steps * decay_limit)))
        else:
            curr_std = 0
        noise = curr_std * torch.randn_like(I_meas_input)
        I_noisy = torch.clamp(I_meas_input + noise, min=0.0)

        phase_pred = net(I_noisy)
        U0_pred = torch.exp(1j * phase_pred.squeeze())
        Ud_pred = fresnel_propagate(U0_pred, H_fwd)
        I_pred = torch.abs(Ud_pred) ** 2

        # --- 1. 计算基础 MSE Loss ---
        mse_loss = F.mse_loss(I_pred, I_meas)

        # --- 2. 计算动态 TV 权重 ---
        current_tv_weight = tv_weight * (0.001 ** ((step / (num_steps - 1)) ** 2))

        # --- 3. 计算 TV Loss 并相加 ---
        tv_loss_val = calculate_tv_loss(phase_pred / (2 * np.pi))  # 归一化计算
        loss = mse_loss + current_tv_weight * tv_loss_val

        # --- 4. 优化与打印 ---
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪：防止突跳
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        phys_losses.append(loss.item())
        if step % 500 == 0:
            torch.cuda.empty_cache()
            curr_lr = optimizer.param_groups[0]['lr']
            print(
                f'[Step {step:05d}] Loss: {loss.item():.4e}  | LR: {curr_lr:.2e} | TV_Wt: {current_tv_weight:.2e}')

    net.eval()
    with torch.no_grad():
        phase_final = net(I_meas_input).squeeze().cpu().numpy()

    del net, optimizer, scheduler,  I_noisy, phase_pred, U0_pred, Ud_pred, I_pred
    torch.cuda.empty_cache()

    # --- GS 算法对比 ---
    # -------- 多平面 z 设置（论文对齐）--------
    num_planes = 3
    z_list = torch.linspace(
        z * 0.8, z * 1.2, num_planes, device=device
    )

    H_fwd_list = [fresnel_kernel(zi, N) for zi in z_list]
    H_bwd_list = [torch.conj(H) for H in H_fwd_list]

    print('Start GS Algorithm...')

    # -------- 多平面幅度测量 --------
    amp_meas_list = []

    with torch.no_grad():
        for zi, Hzi in zip(z_list, H_fwd_list):
            Ud_zi = fresnel_propagate(U0, Hzi)
            I_zi_clean = torch.abs(Ud_zi) ** 2

            # --- 为每个平面添加独立的噪声 ---
            noise_zi = test_noise_std * torch.randn_like(I_zi_clean)
            I_zi_noisy = torch.clamp(I_zi_clean + noise_zi, min=0.0)

            amp_meas_list.append(torch.sqrt(I_zi_noisy + 1e-12))

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
            print(f'[GS-multiZ-noPad] {i:05d} | MSE = {mse_tmp:.4e}')

    # -------------------------------
    # 输出相位
    # -------------------------------
    phase_gs = torch.angle(U0_gs).cpu().numpy()
    del U0_gs, U, Ud, H_fwd_list, H_bwd_list, amp_meas_list, phase_gt, U0, Ud_zi
    torch.cuda.empty_cache()

    # 指标计算与可视化
    mse_p, psnr_p, ssim_p, rec_p = calculate_metrics(phase_final, phase_gt_np)
    mse_g, psnr_g, ssim_g, rec_g = calculate_metrics(phase_gs, phase_gt_np)
    print('\n========== Metrics (Aligned) ==========')
    print(f'PI-UNet   | PSNR: {psnr_p:.2f}, SSIM: {ssim_p:.3f}')
    print(f'GS        | PSNR: {psnr_g:.2f}, SSIM: {ssim_g:.3f}')

    plt.close('all')
    # 绘图
    fig = plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    gt_phase = plt.imshow(phase_gt_np / phase_range, cmap='gray', vmin=phase_min, vmax=phase_max)
    cbar_gt = plt.colorbar(gt_phase, fraction=0.046, pad=0.04)
    cbar_gt.set_label(r'$(2\pi \ rad)$')
    plt.axis("off")
    plt.title('GT Phase')

    plt.subplot(2, 3, 2)
    plt.imshow(I_meas.cpu().numpy(), cmap='magma')
    plt.axis("off")
    plt.title(f'Measured Noisy Intensity (z={z * 1000:.1f}mm)')

    plt.subplot(2, 3, 3)
    plt.plot(phys_losses)
    plt.yscale('log')
    plt.title('PI-UNet Loss')

    plt.subplot(2, 3, 4)
    # plt.imshow(rec_p, cmap='gray')
    phy = plt.imshow(rec_p / phase_range, cmap='gray', vmin=phase_min, vmax=phase_max)
    cbar_phy = plt.colorbar(phy, fraction=0.046, pad=0.04)
    cbar_phy.set_label(r'$(2\pi \ rad)$')
    plt.axis("off")
    plt.title(f'PI-UNet (PSNR: {psnr_p:.2f}  SSIM: {ssim_p:.3f})')

    plt.subplot(2, 3, 5)
    gs = plt.imshow(rec_g / phase_range, cmap='gray', vmin=phase_min, vmax=phase_max)
    cbar_gs = plt.colorbar(gs, fraction=0.046, pad=0.04)
    cbar_gs.set_label(r'$(2\pi \ rad)$')
    plt.axis("off")
    plt.title(f'GS (PSNR: {psnr_g:.2f}  SSIM: {ssim_g:.3f})')

    plt.subplot(2, 3, 6)
    # plt.imshow(rec_p - phase_gt_np, cmap='seismic')
    # 计算误差
    error = rec_p - phase_gt_np
    # 这样 0 是白色，红色是偏大，蓝色是偏小
    v_limit = max(abs(error.min()), abs(error.max()))
    plt.imshow(error, cmap='seismic', vmin=-v_limit, vmax=v_limit)
    plt.colorbar()
    plt.axis("off")
    plt.title('Error Map')

    plt.tight_layout()
    torch.cuda.empty_cache()
    plt.show()
    plt.close(fig)
    del phase_final, phase_gs, rec_p, rec_g, phase_gt_np
    torch.cuda.empty_cache()
    import gc
    gc.collect()
