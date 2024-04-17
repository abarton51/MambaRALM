import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

mamba_acc = np.array([44.6, 41.6, 43.2, 44.5, 44.0, 43.8, 44.0, 38.8, 35.2, 29.6, 21.5], np.float32).reshape(-1, 1)
k = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], np.int16)

##########################################################################
# Interpolate the data
f = interpolate.interp1d(k, mamba_acc.flatten(), kind='quadratic')

# Generate finer k values for smooth plotting
k_interp = np.linspace(k.min(), k.max(), 100)

# Plot original data
plt.scatter(k, mamba_acc, label='Original Data')

# Plot interpolated data
plt.plot(k_interp, f(k_interp), label='Interpolated Data', color='red')

plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Mamba-Chat-2.8B Results vs. k retrieved documents')
plt.legend()
plt.grid(True)
plt.show()
plt.close()
##########################################################################

##########################################################################

# B-spline method

mamba_acc = np.array([44.6, 41.6, 43.2, 44.5, 44.0, 43.8, 44.0, 38.8, 35.2, 29.6, 21.5], np.float32).reshape(-1, 1)
k = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], np.int16)

best_val_score = -1
max_degree = 12
best_degree = -1
for i in range(2, len(k) // 2):
    for j in range(1, max_degree + 1):
        interp = make_pipeline(SplineTransformer(i, degree=j), Ridge(alpha=1e-6))
        interp.fit(k[:,None], mamba_acc)
        if interp.score(k[:,None], mamba_acc) > best_val_score:
            best_interp = interp
            best_degree = j
            best_val_score = best_interp.score(k[:,None], mamba_acc)

print(best_degree)

# Plot results with interpolation line
plt.style.use('seaborn')

# Plot interpolation
plt.plot(k, best_interp.predict(k[:,None]), color='red', label='4-Spline Interpolation')

# Plot original data
orig_mr = np.array([44.6, 41.6, 43.2, 44.5, 44.0, 43.8, 44.0, 38.8, 35.2, 29.6, 21.5], np.float32).reshape(-1, 1)
plt.scatter(k, orig_mr, label='Mamba-Chat Results')

# Format plot
plt.xlabel('k (number of retrieved documents)')
plt.ylabel('% Accuracy')
plt.title('Mamba-Chat-2.8B Accuracy vs k Retrieved Documents', fontsize=18)
plt.xticks(np.arange(20))
plt.yticks(np.array([20, 25, 30, 35, 40, 45, 50], np.float32))
plt.legend()
plt.grid(False)
plt.savefig('mamba-chat-results-over-k.png', dpi=400)
plt.show()