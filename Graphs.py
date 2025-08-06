import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# --- Data from your graphs ---
# You can replace this with your actual data source if needed
sample_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Data for lpg-zeno
lpg_zeno_tabpfn = [0.80, 0.68, 0.54, 0.13, -0.22, -0.49, -0.67, -0.83, -0.91]
lpg_zeno_distnet = [1.04, 0.58, 0.54, 0.10, -0.35, -0.65, -0.74, -0.79, -0.81]

# Data for yalsat-qcp
yalsat_qcp_tabpfn = [5.45, 3.81, 1.56, 0.92, 0.13, -0.42, -0.60, -0.68, -0.69]
yalsat_qcp_distnet = [6.31, 1.63, 1.40, 0.86, 0.18, -0.46, -0.66, -0.70, -0.72]

# --- Plotting ---
# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: lpg-zeno ---
ax1.plot(sample_sizes, lpg_zeno_tabpfn, marker='o', linestyle='-', color='orange', label='TabPFN NLL')
ax1.plot(sample_sizes, lpg_zeno_distnet, marker='o', linestyle='-', color='green', label='DistNet NLL')

# Add text labels for each point, adjusting position to avoid overlap
for x, y in zip(sample_sizes, lpg_zeno_tabpfn):
    # Place TabPFN labels slightly ABOVE the point
    ax1.text(x, y + 0.05, f'{y:.2f}', ha='center', va='bottom', color='orange', fontsize=9)
for x, y in zip(sample_sizes, lpg_zeno_distnet):
    # Place DistNet labels slightly BELOW the point
    ax1.text(x, y - 0.05, f'{y:.2f}', ha='center', va='top', color='green', fontsize=9)

ax1.set_xscale('log')
ax1.set_title('lpg-zeno')
ax1.set_xlabel('Sample Size')
ax1.set_ylabel('NLL (Lower is Better)')
ax1.grid(True, which="both", ls="--", linewidth=0.5)
ax1.legend()
ax1.set_xticks(sample_sizes)
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")


# --- Plot 2: yalsat-qcp ---
ax2.plot(sample_sizes, yalsat_qcp_tabpfn, marker='o', linestyle='-', color='orange', label='TabPFN NLL')
ax2.plot(sample_sizes, yalsat_qcp_distnet, marker='o', linestyle='-', color='green', label='DistNet NLL')

# Add text labels for each point, adjusting position to avoid overlap
for x, y in zip(sample_sizes, yalsat_qcp_tabpfn):
    # Place TabPFN labels slightly ABOVE the point
    ax2.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='orange', fontsize=9)
for x, y in zip(sample_sizes, yalsat_qcp_distnet):
    # Place DistNet labels slightly BELOW the point
    ax2.text(x, y - 0.1, f'{y:.2f}', ha='center', va='top', color='green', fontsize=9)

ax2.set_xscale('log')
ax2.set_title('yalsat-qcp')
ax2.set_xlabel('Sample Size')
ax2.grid(True, which="both", ls="--", linewidth=0.5)
ax2.legend()
ax2.set_xticks(sample_sizes)
ax2.get_xaxis().set_major_formatter(ScalarFormatter())
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")


# --- Final Adjustments ---
plt.tight_layout(pad=2.0)
fig.suptitle('TabPFN vs DistNet: NLL vs Sample Size', fontsize=16)
fig.subplots_adjust(top=0.85)

# Show the plot
plt.show()

# To save the figure to a file
# fig.savefig('combined_plot_with_labels.png', dpi=300)