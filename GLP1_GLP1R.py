
import numpy as np
import matplotlib.pyplot as plt

# Calculate fraction of activated GLP1 receptors
def Fact(G, S, Kd1, Kd2):
    return (Kd2 * G + Kd1 * S) / (
        Kd1 * Kd2 + Kd2 * G + Kd1 * S
    )

# Dissociation constants (nM)
Kd1 = 0.17  # GLP-1
Kd2 = 0.38  # Semaglutide

# GLP-1 concentrations (nM)
G_fasting = 0.0150
G_fed = 0.040

# Semaglutide concentration range (nM)
S = np.logspace(-3, 3, 500)

# Calculate receptor activation
response_fasting = Fact(G_fasting, S, Kd1, Kd2)
response_fed = Fact(G_fed, S, Kd1, Kd2)

# Create concentration-response plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogx(
    S, response_fasting,
    linewidth=2,
    label="Fasting GLP-1 (0.015 nM)"
)

ax.semilogx(
    S, response_fed,
    linewidth=2,
    label="After-meal GLP-1 (0.040 nM)"
)

ax.axvline(
    1.0,
    color="gray",
    linestyle="--",
    label="Semaglutide reference (1 nM)"
)

ax.set_xlabel("Semaglutide concentration (nM)")
ax.set_ylabel("Fraction of active GLP1R")
ax.set_title("GLP-1 and Semaglutide Co-activation of GLP1R")

ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()

# Save graph in the same folder as this Python file
import os
folder = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(folder, "GLP1_GLP1R.png")

fig.savefig(save_path, dpi=300, bbox_inches="tight")

# Display graph
plt.show(block=True)
