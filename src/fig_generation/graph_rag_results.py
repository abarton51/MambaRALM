import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from scipy.interpolate import CubicSpline

mamba_acc = np.array([44.6, 41.6, 43.2, 44.5, 44.0, 43.8, 44.0, 38.8, 35.2, 29.6], np.float16).reshape(-1, 1)
dolly_acc = np.array([43.2, 44.9, 45.8, 43.9, 46.3, 47.0, 47.8, 43.6, 29.8, 10.7], np.float16).reshape(-1, 1)
k = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.int16)

##########################################################################
# Interpolate the data
f = interpolate.interp1d(k, mamba_acc.flatten(), kind='quadratic')

# Generate finer k values for smooth plotting
k_interp = np.linspace(k.min(), k.max(), 100)

# Plot original data
plt.scatter(k, mamba_acc, label='Mamba % Accuracy Results')
plt.scatter(k, dolly_acc, label='Dolly % Accuracy Result')

# Plot interpolated data
plt.plot(k_interp, f(k_interp), label='Interpolated Data', color='red')

plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Mamba-Chat-2.8B/Dolly-v2-3B Results vs. k retrieved documents')
plt.legend()
plt.grid(True)
plt.show()
plt.close()
##########################################################################
# Cubic Splines

x = k
y = mamba_acc
spl = CubicSpline(x, y)

fig, ax = plt.subplots(4, 1, figsize=(5, 7))
xnew = np.linspace(0, 10, num=1001)
ax[0].plot(xnew, spl(xnew))
ax[0].plot(x, y, 'o', label='data')
ax[1].plot(xnew, spl(xnew, nu=1), '--', label='1st derivative')
plt.tight_layout()
plt.show()
plt.close()
##########################################################################

# B-spline method

mamba_acc = np.array([44.6, 41.6, 43.2, 44.5, 44.0, 43.8, 44.0, 38.8, 35.2, 29.6], np.float16).reshape(-1, 1)
dolly_acc = np.array([43.2, 44.9, 45.8, 43.9, 46.3, 47.0, 47.8, 43.6, 29.8, 10.7], np.float16).reshape(-1, 1)

k = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.int16)

def interp_func(X, max_degree):
    best_val_score = -1
    best_degree = -1
    for i in range(2, len(k) // 2):
        for j in range(1, max_degree + 1):
            interp = make_pipeline(SplineTransformer(i, degree=j), Ridge(alpha=1e-6))
            interp.fit(k[:,None], X)
            if interp.score(k[:,None], X) > best_val_score:
                best_interp = interp
                best_degree = j
                best_val_score = best_interp.score(k[:,None], X)
                
    return best_interp, best_degree

mamba_interp, mamba_deg = interp_func(mamba_acc, 4)
dolly_interp, dolly_deg = interp_func(dolly_acc, 4)
print(f"Mamba interp deg: {mamba_deg}; Dolly interp deg: {dolly_deg}")

# Plot results with interpolation line
plt.style.use('seaborn')

# Plot interpolation
plt.plot(k_interp, mamba_interp.predict(k_interp[:,None]), color='purple', label=f'Mamba: {mamba_deg}-Spline Interpolation')
plt.plot(k_interp, dolly_interp.predict(k_interp[:,None]), color='orange', label=f'Dolly: {dolly_deg}-Spline Interpolation')

# Plot original data
orig_mr = np.array([44.6, 41.6, 43.2, 44.5, 44.0, 43.8, 44.0, 38.8, 35.2, 29.6], np.float16).reshape(-1, 1)
orig_dr = np.array([43.2, 44.9, 45.8, 43.9, 46.3, 47.0, 47.8, 43.6, 29.8, 10.7], np.float16).reshape(-1, 1)
plt.scatter(k, orig_mr, label='Mamba-Chat Results')
plt.scatter(k, orig_dr, label='Dolly Results', color='red')

# Format plot
plt.xlabel('k (number of retrieved chunks)')
plt.ylabel('% Accuracy')
plt.title('Mamba-Chat-2.8B/Dolly-v2-3B % Accuracy vs k Retrieved Chunks', fontsize=16)
plt.xticks(np.arange(10))
plt.yticks(np.array([20, 25, 30, 35, 40, 45, 50], np.float32))
plt.legend()
plt.grid(False)
plt.savefig('rag-results-over-k.png', dpi=400)
plt.show()