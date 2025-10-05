import pandas as pd
from dowhy import CausalModel

# Load synthetic data
data = pd.read_csv('synthetic_causal_data.csv')

# Define causal model
causal_model = CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome',
    common_causes='confounder'

    #confounder → treatment
    #confounder → outcome
    #treatment → outcome

)

# Identify causal effect
identified_estimand = causal_model.identify_effect()
print("Identified Estimand:", identified_estimand)

# (Optional) Estimate effect - use Colab for heavy computation later
# estimate = causal_model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
# print("Estimate:", estimate.value)