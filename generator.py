import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置 GUI 后端
matplotlib.use("TkAgg")

# ===============================
# 参数设置
# ===============================
input_image_path = "./database/image4.png"
output_npy_path = "./database/output4.npy"
output_png_path = "./database/output4.png"
phase_range = 2 * np.pi

# 1. 读取并基础预处理
img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("无法读取输入图像")
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

# ===============================
# 极简逻辑：仅保留 CLAHE 和 留白
# ===============================

# CLAHE 调整对比度
# 如果觉得亮点还是突兀，调小 clipLimit (比如 1.2)；如果想更清晰，调大 (比如 3.0)
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
img_processed = clahe.apply(img).astype(np.float32)

# def sigmoid_mapping(x, cutoff=0.5, gain=5):
#     x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
#     return 1 / (1 + np.exp(-gain * (x_norm - cutoff)))
#
# 
# img_sigmoid = sigmoid_mapping(img_processed, cutoff=0.4, gain=4)

# 最直接的线性归一化到 [0, 1]
img_norm = (img_processed - img_processed.min()) / (img_processed.max() - img_processed.min() + 1e-8)

# [留白策略]：相位量程控制
phase_min, phase_max = 0, 0.5
phase = (img_norm * (phase_max - phase_min) + phase_min) * phase_range

# [保边滤波]：防止像素级突跳（噪声）
img_final = cv2.bilateralFilter(phase.astype(np.float32), d=3, sigmaColor=0.1, sigmaSpace=0.8)

# ===============================
# 2. 保存输出
# ===============================
# 保存高精度 NPY 供网络使用
np.save(output_npy_path, img_final.astype(np.float32))

# 保存可视化 PNG (16-bit)
phase_16bit = ((img_final / phase_range) * 65535).astype(np.uint16)
cv2.imwrite(output_png_path, phase_16bit)

print(f"数据已保存：{output_npy_path} 和 {output_png_path}")

# ===============================
# 3. 可视化 (完美适配数据范围)
# ===============================
plt.figure(figsize=(6, 5))

# 使用相位的实际 min/max 作为显示区间
im = plt.imshow(img_final / phase_range, cmap='gray', vmin=0, vmax=0.5)

cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label(r'$(2\pi \ rad)$')
plt.title("Phase Map")
plt.axis("off")
plt.show()
