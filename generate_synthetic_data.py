import pandas as pd
import numpy as np

# Generate synthetic data (100 rows)
np.random.seed(42)
n_samples = 200
T = np.random.binomial(1, 0.5, n_samples)  # Treatment (0 or 1)
C = np.random.normal(0, 1, n_samples)      # Confounder
Y = 2 * T + C + np.random.normal(0, 0.1, n_samples)  # Outcome

# Create DataFrame
synthetic_data = pd.DataFrame({'treatment': T, 'confounder': C, 'outcome': Y})
synthetic_data.to_csv('synthetic_causal_data.csv', index=False)

print("Generated synthetic_causal_data.csv with 200 rows.")