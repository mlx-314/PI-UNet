import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =========================
# 1. 参数（对齐作者）
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 440
wavelength = 632.8e-9
pixel_size = 8e-6
z = 22.3e-3

num_steps = 10001
lr = 1e-2
noise_std_init = 0.01
tv_weight = 5e-8

# =========================
# 2. 读取作者衍射图
# =========================
img = cv2.imread('./database/diff_1.tif', -1).astype(np.float32)

# ⭐严格按作者裁剪
img = img[280:280 + N, 260:260 + N]

# ⭐归一化（关键）
img = img / np.max(img)

I_meas = torch.tensor(img, device=device)
I_meas_input = I_meas.unsqueeze(0).unsqueeze(0)


# =========================
# 3. FFT SHIFT（对齐作者）
# =========================


def fftshift(x):
    return torch.fft.fftshift(x)


def ifftshift(x):
    return torch.fft.ifftshift(x)


# =========================
# 4. Angular Spectrum传播（核心替换）
# =========================


def angular_spectrum(phase):
    M = N
    L = N * pixel_size
    # 强制去掉直流分量（均值），防止数值漂移
    phase = phase - torch.mean(phase)
    # phase → complex64
    U_in = torch.exp(1j * phase.to(torch.complex64))

    U_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U_in)))

    fx = np.linspace(-M / 2, M / 2 - 1, M) / L
    FX, FY = np.meshgrid(fx, fx)

    FX = torch.tensor(FX, device=device, dtype=torch.float32)
    FY = torch.tensor(FY, device=device, dtype=torch.float32)

    k = 2 * np.pi / wavelength

    inside = 1 - (wavelength ** 2) * (FX ** 2 + FY ** 2)
    inside = torch.clamp(inside, min=0.0)

    H = torch.sqrt(inside).to(torch.complex64)

    H = torch.exp(1j * k * z * H)

    U_fft = U_fft * H

    U_out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(U_fft)))

    In = torch.abs(U_out) ** 2

    In = In / (torch.max(In) + 1e-8)

    return In


def forward_prop(U):
    M = N
    L = N * pixel_size

    U_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U)))

    fx = np.linspace(-M / 2, M / 2 - 1, M) / L
    FX, FY = np.meshgrid(fx, fx)

    FX = torch.tensor(FX, device=device, dtype=torch.float32)
    FY = torch.tensor(FY, device=device, dtype=torch.float32)

    k = 2 * np.pi / wavelength

    inside = 1 - (wavelength ** 2) * (FX ** 2 + FY ** 2)
    inside = torch.clamp(inside, min=0.0)

    H = torch.exp(1j * k * z * torch.sqrt(inside).to(torch.complex64))

    U_fft = U_fft * H
    U_out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(U_fft)))

    return U_out


def backward_prop(U):
    M = N
    L = N * pixel_size

    U_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U)))

    fx = np.linspace(-M / 2, M / 2 - 1, M) / L
    FX, FY = np.meshgrid(fx, fx)

    FX = torch.tensor(FX, device=device, dtype=torch.float32)
    FY = torch.tensor(FY, device=device, dtype=torch.float32)

    k = 2 * np.pi / wavelength

    inside = 1 - (wavelength ** 2) * (FX ** 2 + FY ** 2)
    inside = torch.clamp(inside, min=0.0)

    H = torch.exp(1j * k * z * torch.sqrt(inside).to(torch.complex64))

    U_fft = U_fft * torch.conj(H)

    U_out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(U_fft)))

    return U_out
# =========================
# 5. TV Loss
# =========================


def tv_loss(Img):
    return torch.sum(torch.abs(Img[:, :, 1:, :] - Img[:, :, :-1, :])) + \
        torch.sum(torch.abs(Img[:, :, :, 1:] - Img[:, :, :, :-1]))


# =========================
# 6. 网络（保持你原来的）
# =========================
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
        # raw = self.out_conv(d1)
        #
        # # 强制去均值
        # # 每一轮都把相位拉回到平均值为 0.3 * 2pi 的地方
        # phase = raw - torch.mean(raw) + (0.3 * 2 * np.pi)
        #
        # return F.relu(phase)
        # # return F.relu(self.out_conv(d1))


# =========================
# 7. 训练
# =========================


net = DIP_UNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=8e-5)

losses = []

print("Start optimization...")

for step in range(num_steps):

    # 噪声退火（保持你原来的）
    decay_limit = 0.95
    if step < num_steps * decay_limit:
        curr_std = noise_std_init * (1e-4 ** (step / (num_steps * decay_limit)))
    else:
        curr_std = 0

    noise = curr_std * torch.randn_like(I_meas_input)
    I_in = torch.clamp(I_meas_input + noise, min=0)

    phase_pred = net(I_in)

    I_pred = angular_spectrum(phase_pred.squeeze())

    loss_mse = F.mse_loss(I_pred, I_meas)

    current_tv = tv_weight * (0.001 ** ((step / num_steps) ** 2))
    # # 增加一个范围约束 Loss
    # # 假设我们期望相位在 0 到 0.6 * 2pi 之间
    # loss_range = torch.mean(F.relu(phase_pred - 0.6 * 2 * np.pi) ** 2) + \
    #     torch.mean(F.relu(-phase_pred) ** 2)

    # 总 Loss 调整为：
    loss = loss_mse + current_tv * tv_loss(phase_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if step % 500 == 0:
        print(f"[{step}] Loss: {loss.item():.4e}")
        # print(f"Phase Mean: {phase_pred.mean().item():.2f}, Phase Max: {phase_pred.max().item():.2f}")

# =========================
# 8. 最终结果
# =========================
net.eval()
with torch.no_grad():
    phase_final = net(I_meas_input).squeeze()
    I_final = angular_spectrum(phase_final).cpu().numpy()
    phase_final = phase_final.cpu().numpy()

I_input_np = I_meas.cpu().numpy()
# =========================
# 8.5 GS算法（单幅）
# =========================
print('Start GS...')

amp = torch.sqrt(I_meas + 1e-12)

phase_gs_t = torch.zeros((N, N), device=device)

num_gs_iters = 1201

for it in range(num_gs_iters):

    # 物面：纯相位
    U0 = torch.exp(1j * phase_gs_t)

    # 正传播
    Ud = forward_prop(U0)

    # 使预测场 Ud 的幅值范围与测量场 amp 处于同一量级
    with torch.no_grad():
        current_max = torch.max(torch.abs(Ud))
        target_max = torch.max(amp)
        Ud = Ud * (target_max / (current_max + 1e-8))

    if it % 200 == 0:
        gs_mse = F.mse_loss(torch.abs(Ud)**2, I_meas).item()
        print(f'[GS Iter {it:04d}] Intensity MSE: {gs_mse:.4e}')
    # --- 替换振幅约束 ---
    Ud = amp * torch.exp(1j * torch.angle(Ud))

    # --- 反传播回物面 ---
    U0_back = backward_prop(Ud)

    # --- 更新相位
    new_phase = torch.angle(U0_back)
    # # 移除直流分量，防止全局相位漂移
    # phase_gs_t = new_phase - torch.mean(new_phase)
    # 让最小值为0
    # phase_gs_t = new_phase - torch.min(new_phase)
    # # --- 改进点：在相位空间引入平滑约束 ---
    # new_phase = torch.angle(U0)
    #
    # # 每 10 次迭代进行一次微弱的 TV 降噪
    # if it % 10 == 0:
    #     # 这里可以使用一个简单的 3x3 平均卷积来抑制高频噪声
    #     kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
    #     new_phase = F.conv2d(new_phase.view(1, 1, N, N), kernel, padding=1).squeeze()
    #
    phase_gs_t = new_phase


phase_gs = phase_gs_t.cpu().numpy()

with torch.no_grad():
    I_gs = angular_spectrum(torch.tensor(phase_gs, device=device)).cpu().numpy()

# =========================
# 9. 指标（强度）
# =========================
psnr_p = peak_signal_noise_ratio(I_input_np, I_final, data_range=1)
ssim_p = structural_similarity(I_input_np, I_final, data_range=1)

psnr_g = peak_signal_noise_ratio(I_input_np, I_gs, data_range=1)
ssim_g = structural_similarity(I_input_np, I_gs, data_range=1)

print("\n===== Metrics =====")
print(f"PI-UNet | PSNR: {psnr_p:.2f}, SSIM: {ssim_p:.4f}")
print(f"GS      | PSNR: {psnr_g:.2f}, SSIM: {ssim_g:.4f}")

# =========================
# 10. 可视化
# =========================
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(I_input_np, cmap='magma')
plt.title("Input Diffraction")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(I_final, cmap='magma')
plt.title(f"PI-UNet Diff")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.plot(losses)
plt.yscale('log')
plt.title("Loss")

plt.subplot(2, 3, 4)
plt.imshow(phase_final, cmap='gray')
plt.title(f"PI-UNet Phase\nPSNR:{psnr_p:.2f} SSIM:{ssim_p:.3f}")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(phase_gs, cmap='gray')
plt.title(f"GS Phase\nPSNR:{psnr_g:.2f} SSIM:{ssim_g:.3f}")
plt.axis('off')

plt.subplot(2, 3, 6)
error = I_final - I_input_np
v = np.max(np.abs(error))
plt.imshow(error, cmap='seismic', vmin=-v, vmax=v)
plt.title("Intensity Error Map")
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()
torch.cuda.empty_cache()
