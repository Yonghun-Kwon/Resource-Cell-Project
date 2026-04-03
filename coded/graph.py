import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": ["Times New Roman"],
})

# === 프레임워크 레이블 (x축) ===
frameworks = [
    "PyTorch",
    "PyTorch\nMobile",
    "LibTorch\nC++",
    "LibTorch C++\n(Mobile)",
    "TensorFlow",
    "TensorFlow\nLite",
    "TensorFlow\nLite C++"
]
x = np.arange(len(frameworks))
width = 0.7  # 바 너비

# === Set 1: MobileNet-V2 (7개 프레임워크 값) ===
mobilenet_values = np.array([82.08, 76.78, 892.24, 81.26, 148.45, 12.07, 10.9])

# === Set 2: ResNet-50 (7개 프레임워크 값) ===
resnet_values = np.array([809.23, 217.41, 368.7, 218.77, 334.64, 103.167, 102.415])

# =========================================
# 2×1 Subplot
# =========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fig.subplots_adjust(hspace=0.3, wspace=0.35)

titles = ["(a) MobileNet-V2", "(b) ResNet-50"]

# 색상
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

# -------- Draw each subplot --------
axes[0].bar(x, mobilenet_values, width, color=colors)
axes[1].bar(x, resnet_values, width, color=colors)

for i, ax in enumerate(axes):
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, fontsize=9, rotation=0)
    ax.set_ylabel("Inference Time (ms)", fontsize=12)
    ax.grid(axis="y", linestyle="dotted", alpha=0.6)

    # 제목을 아래로 이동
    ax.text(0.5, -0.20, titles[i],
            ha='center', va='top',
            transform=ax.transAxes, fontsize=14, weight='bold')

plt.tight_layout()
plt.show()