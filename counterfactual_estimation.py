import pandas as pd
import dowhy
from dowhy import CausalModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load or regenerate data
data = pd.read_csv('synthetic_causal_data.csv')
treatment_counts = data['treatment'].value_counts()
print("Initial treatment balance:", treatment_counts)

# Regenerate if severely imbalanced (less than 30 in any group), with retry
max_retries = 3
for retry in range(max_retries):
    if treatment_counts.min() < 30:
        print(f"Retry {retry + 1}/{max_retries}: Regenerating balanced dataset...")
        n_samples = 200
        data = pd.DataFrame({
            'treatment': np.random.binomial(1, 0.5, n_samples),
            'confounder': np.random.normal(0, 1.5, n_samples),
            'outcome': 2 * np.random.binomial(1, 0.5, n_samples) + np.random.normal(0, 1.5, n_samples) + np.random.normal(0, 0.1, n_samples)
        })
        data.to_csv('synthetic_causal_data.csv', index=False)
        treatment_counts = data['treatment'].value_counts()
        print(f"New treatment balance after retry {retry + 1}:", treatment_counts)
    else:
        break
else:
    print("Max retries reached; using imbalanced data with warning")
    treatment_counts = data['treatment'].value_counts()

# Plot confounder distribution
plt.hist(data[data['treatment'] == 1]['confounder'], bins=20, alpha=0.5, label='Treated')
plt.hist(data[data['treatment'] == 0]['confounder'], bins=20, alpha=0.5, label='Untreated')
plt.title('Confounder Distribution by Treatment')
plt.xlabel('Confounder')
plt.ylabel('Count')
plt.legend()
plt.savefig('confounder_distribution.png')
plt.show()

# Define SCM
causal_model = CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome',
    common_causes=['confounder']
)

# Identify causal effect
identified_estimand = causal_model.identify_effect()

# Counterfactual estimation (Linear Regression)
units = [5, 15, 25, 35, 45]
counterfactuals_lr = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    target_units=units
)
print("Counterfactuals (Linear Regression):", counterfactuals_lr.value)

# Counterfactual estimation with balance handling
try:
    # Calculate imbalance metrics
    treatment_ratio = treatment_counts.max() / treatment_counts.min()
    print(f"Treatment ratio (max/min): {treatment_ratio:.2f}")

    if treatment_counts.min() < 10:
        raise ValueError("Severe imbalance: less than 10 samples in one group")

    if treatment_ratio > 3:
        # Use IPW for high imbalance
        print("Using IPW due to high imbalance ratio")
        counterfactuals_psm = causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_weighting",
            target_units=units,
            method_params={
                "weighted_regression": True,
                "stabilized": True
            }
        )
    else:
        # Use PSM with optimal parameters for moderate imbalance
        print("Using PSM with balanced parameters")
        counterfactuals_psm = causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_matching",
            target_units=units,
            method_params={
                "n_neighbors": min(5, treatment_counts.min() // 10),
                "caliper": 0.2,
                "replace": True
            }
        )
    
    print("Counterfactual estimation successful:", counterfactuals_psm.value)

except Exception as e:
    print(f"Primary estimation failed: {str(e)}")
    
    # Fallback strategy with stratification
    try:
        print("Attempting fallback with stratification...")
        counterfactuals_psm = causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_stratification",
            target_units=units[:3],  # Use first 3 of the new units [5, 15, 25]
            method_params={
                "n_strata": 3,
                "min_samples_per_stratum": 5
            }
        )
        print("Fallback estimation successful")
    except Exception as e2:
        print(f"Fallback estimation failed: {str(e2)}")
        counterfactuals_psm = None

# Save results
with open('counterfactual_results.txt', 'w') as f:
    f.write(f"Counterfactuals (Linear Regression): {counterfactuals_lr.value}\n")
    if counterfactuals_psm:
        f.write(f"Counterfactuals (Propensity Score Matching): {counterfactuals_psm.value}\n")
    else:
        f.write("Propensity Score Matching: Failed due to imbalance or insufficient matches\n")

# Fit linear model to estimate coefficients
X = data[['treatment', 'confounder']]
y = data['outcome']
model = LinearRegression()
model.fit(X, y)
alpha = model.intercept_
beta = model.coef_[0]
gamma = model.coef_[1]
print(f"Fitted coefficients: alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")

# Calculate residuals
data['predicted_outcome'] = model.predict(X)
data['residual'] = data['outcome'] - data['predicted_outcome']

# Manual counterfactual for unit 15 with residual
unit_15 = data.iloc[15]
actual_treatment = unit_15['treatment']
actual_outcome = unit_15['outcome']
confounder = unit_15['confounder']
residual = unit_15['residual']
counterfactual_treatment = 1 if actual_treatment == 0 else 0
counterfactual_outcome = alpha + beta * counterfactual_treatment + gamma * confounder + residual
print(f"Unit 15: Actual Outcome={actual_outcome}, Counterfactual Outcome (with residual)={counterfactual_outcome:.4f}")

# Experiment: Add stochastic noise to counterfactual
stochastic_noise = np.random.normal(0, 0.1)
counterfactual_outcome_noisy = counterfactual_outcome + stochastic_noise
print(f"Unit 15: Counterfactual Outcome (with stochastic noise)={counterfactual_outcome_noisy:.4f}")