import numpy as np
import matplotlib.pyplot as plt

def yeole_ratio(N, D):
    return (N * D) / (N + D)

def M_opt(K, D):
    if K <= 1:
        return D
    elif abs(K - np.floor(K)) < 1e-9:  # check if K is integer
        return int(np.floor(K) - 1)
    else:
        return int(np.floor(K))

def L_R_opt(K, D):
    M = M_opt(K, D)
    return 1 - M / D

# Parameters
D_values = [10, 50, 100, 500]  # test for various D
N_range = np.arange(1, 5000)   # vary N

plt.figure(figsize=(6, 5))

for D in D_values:
    K_vals = yeole_ratio(N_range, D)
    L_vals = [L_R_opt(K, D) for K in K_vals]

    # plot curve with small markers
    plt.plot(K_vals, L_vals, label=f"D={D}", marker=".", markersize=2, linewidth=1)

plt.xlabel("Yeole Ratio K")
plt.ylabel(r"$L^{R}_{opt}$")
plt.title(r"Optimal Relative Information Loss $L^{R}_{opt}$ vs Yeole Ratio $K$")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
