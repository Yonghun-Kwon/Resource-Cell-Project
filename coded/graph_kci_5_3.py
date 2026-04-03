import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": ["Times New Roman"],
})

# ================== Figure & GridSpec ==================
fig = plt.figure(figsize=(6, 6), dpi=150)
gs = gridspec.GridSpec(2, 1)
gs.update(hspace=0.6)

axs = []
axs.append(fig.add_subplot(gs[0, 0]))   # (d)
axs.append(fig.add_subplot(gs[1, 0]))   # (e)

# ======================================================
# (d) Resource-Cell Plot at 10 J
# ======================================================
rc_labels = ['GEMM', 'Convolution', 'Depthwise']
device_labels = ['Raspberry Pi 3B+', 'Raspberry Pi 4B', 'Raspberry Pi 5B']

# ✅ 논문용: 계산된 결과를 직접 명시
predicted_array = np.array([
    [3424.029, 769.2, 67.12],     # Raspberry Pi 3B+
    [6933.527, 1586, 148.917],   # RPi4
    [15446.257, 2886, 384.767],  # RPi5
])

x = np.arange(len(device_labels))
bar_width = 0.2
colors_ops = ["#81bae9","#a5fa93",  "#E9B0B0"]

for op_idx in range(3):
    axs[0].bar(
        x + (op_idx - 1.5) * bar_width,
        predicted_array[:, op_idx],
        width=bar_width,
        label=rc_labels[op_idx],
        color=colors_ops[op_idx]
    )

axs[0].set_xticks(x)
axs[0].set_xticklabels(device_labels)
axs[0].set_ylabel("Resource-Cell", fontsize=10)
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

axs[0].legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.25),
    ncol=4,
    frameon=False,
    fontsize=9
)

axs[0].text(
    0.5, -0.2,
    "(a) Resource-Cells Allowed at 10 J",
    ha='center', va='top',
    transform=axs[0].transAxes,
    fontsize=12,
)

# ======================================================
# (e) Operating Time Plot
# ======================================================
survival_sec = np.array([
    [1234.860, 1471.947141, 1738.918],
    [1162.835769, 1421.997512, 1782.186],
    [815.9675789, 1004.025865, 1293.363],
])

load_labels = ['Full load', 'Half load', '100 Resource-Cells load ']
x2 = np.arange(len(device_labels))
bar_width2 = 0.25
colors_load = ["#7aafdb","#8dd17f",  "#C49797"]

for j in range(3):
    axs[1].bar(
        x2 + (j - 1) * bar_width2,
        survival_sec[:, j],
        width=bar_width2,
        label=load_labels[j],
        color=colors_load[j],
    )

axs[1].set_xticks(x2)
axs[1].set_xticklabels(device_labels)
axs[1].set_ylabel("Operating Time (sec.)", fontsize=10)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

axs[1].legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.25),
    ncol=3,
    frameon=False,
    fontsize=9
)

axs[1].text(
    0.5, -0.2,
    "(b) Operating Time Until Power Depletion at 3600 J",
    ha='center', va='top',
    transform=axs[1].transAxes,
    fontsize=12,
)

plt.tight_layout()
plt.show()
