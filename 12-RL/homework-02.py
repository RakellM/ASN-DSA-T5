# %%
import numpy as np
from sklearn.metrics import r2_score

# Valores reais de y (por exemplo, de um dataset qualquer)
y_true = np.array([5, 7, 10, 6, 8])

# Previsão usando apenas a média de y
y_pred = np.full_like(y_true, fill_value=y_true.mean(), dtype=np.float64)

# Cálculo do R²
r2 = r2_score(y_true, y_pred)

print("Valores reais:", y_true)
print("Previsão (média):", y_pred)
print("R²:", r2)

# %%
