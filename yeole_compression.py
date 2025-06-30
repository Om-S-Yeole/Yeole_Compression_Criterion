import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data[:10000]  # Use first 10,000 samples

N, D = X.shape
S0 = N * D
K = (N * D) / (N + D)
K_floor = int(np.floor(K))

# Memory per float64 in bytes
BYTES_PER_ELEMENT = 8

# Set of M values to test
M_values = [100, 300, 500, 700, K_floor, K_floor + 1, K_floor + 50]

results = []

for M in M_values:
    pca = PCA(n_components=M)
    Z = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(Z)
    
    mse = mean_squared_error(X, X_reconstructed)
    
    # Element-wise storage
    S1 = M * (N + D)
    S1_less_than_S0 = "Yes" if S1 < S0 else "No"
    
    # Actual memory in bytes
    mem_X = X.nbytes
    mem_Z = Z.nbytes
    mem_B = pca.components_.T.nbytes  # B = D x M
    
    mem_total = mem_Z + mem_B
    
    results.append({
        'M': M,
        'S1 = M(N+D)': S1,
        'S1 < S0?': S1_less_than_S0,
        'MSE': round(mse, 6),
        'Memory_Original_MB': round(mem_X / 1e6, 2),
        'Memory_Final_MB': round(mem_total / 1e6, 2),
        'Saved_MB?': "Yes" if mem_total < mem_X else "No"
    })

# Create DataFrame
df = pd.DataFrame(results)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df["M"], df["Memory_Final_MB"], width=15, label='Compressed Memory (MB)', color='skyblue')
plt.axhline(round(mem_X / 1e6, 2), color='red', linestyle='--', label=f'Original Memory ({round(mem_X / 1e6, 2)} MB)')
plt.title('Memory Usage vs Number of Principal Components (M)')
plt.xlabel('Number of Components (M)')
plt.ylabel('Memory Used (MB)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("memory_usage_vs_M.png", dpi=300)
plt.show()

# Pick any digit index to visualize (e.g., 7th image)
index = 7
M_visuals = [100, 500, 727, 777]

fig, axs = plt.subplots(2, len(M_visuals) + 1, figsize=(12, 4))

# Original image
axs[0, 0].imshow(X[index].reshape(28, 28), cmap='gray')
axs[0, 0].set_title("Original")
axs[0, 0].axis('off')

# Reconstruction at different M values
for i, M in enumerate(M_visuals):
    pca = PCA(n_components=M)
    Z = pca.fit_transform(X)
    X_recon = pca.inverse_transform(Z)
    axs[0, i+1].imshow(X_recon[index].reshape(28, 28), cmap='gray')
    axs[0, i+1].set_title(f"M = {M}")
    axs[0, i+1].axis('off')

# Difference images (optional)
for i, M in enumerate(M_visuals):
    pca = PCA(n_components=M)
    Z = pca.fit_transform(X)
    X_recon = pca.inverse_transform(Z)
    diff = np.abs(X[index] - X_recon[index]).reshape(28, 28)
    axs[1, i+1].imshow(diff, cmap='hot')
    axs[1, i+1].set_title(f"Diff M = {M}")
    axs[1, i+1].axis('off')

axs[1, 0].axis('off')  # Empty placeholder

plt.suptitle("Original vs Reconstructed Samples at Different M")
plt.tight_layout()
plt.savefig("reconstruction_comparison.png", dpi=300)
plt.show()