import pandas as pd
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import json
import ast  # for robust parsing of Python-style dicts
from RL_integration import QLearningAgent, CausalEnv

# Helper: compute counterfactual using normalized confounder
def compute_counterfactual(unit_idx, treatment):
    unit = data.iloc[unit_idx]
    # Use normalized confounder (consistent with environment)
    confounder = unit.get('confounder_normalized', unit.get('confounder'))
    # Compute residual using normalized confounder
    residual = unit['outcome'] - (alpha + beta * unit['treatment'] + gamma * confounder)
    counterfactual_outcome = alpha + beta * treatment + gamma * confounder + residual
    return counterfactual_outcome

# Helper: enhanced fallback parser for queries with multiple terminology support
def fallback_parse(query):
    print(f"Processing query: {query}")
    query_l = query.lower()
    
    # Enhanced unit/patient identification
    unit_patterns = [
        r'(?:unit|patient|person|individual|subject|case)\s+(\d+)',  # Various terms for unit
        r'#(\d+)',  # Handle #123 format
        r'number\s+(\d+)',  # "number 123"
        r'(\d+)(?:st|nd|rd|th)?(?:\s+patient|\s+unit)?'  # Handle "1st patient", "2nd unit", etc.
    ]
    
    unit = None
    for pattern in unit_patterns:
        match = re.search(pattern, query_l)
        if match:
            unit = int(match.group(1))
            print(f"Found unit/patient ID: {unit}")
            break
    
    if unit is None:
        print("Could not identify patient/unit number")
        return None
    
    # Enhanced treatment pattern matching
    treatment_match = re.search(r'(?:treatment|medicine|medication|dose|therapy)\s+(\d+)', query_l)
    treatment = int(treatment_match.group(1)) if treatment_match else None
    
    # Enhanced scenario detection with more natural language patterns
    treatment_indicators = [
        "use", "get", "gets", "given", "receive", "take", "takes", "taking",
        "implement", "apply", "start", "begin", "give", "administer", "prescribe",
        "treated", "treatment", "medicine", "medication", "therapy", "dose"
    ]
    
    skip_indicators = [
        "avoid", "skip", "omit", "stop", "without", "no", "don't", "dont",
        "untreated", "not", "never", "refuse", "decline", "deny", "withdraw"
    ]
    
    effect_indicators = [
        "effect", "impact", "outcome", "result", "consequence", "influence",
        "change", "difference", "what happens", "what would happen",
        "how does", "how would", "prediction", "predict", "forecast"
    ]
    
    # Check for treatment scenario with enhanced patterns
    if any(word in query_l for word in treatment_indicators):
        treatment = treatment if treatment is not None else 1
        scenario = "treated" if treatment == 1 else "untreated"
        return {"unit": unit, "treatment": treatment, "scenario": scenario}
    
    # Check for skip/untreated scenario
    if any(word in query_l for word in skip_indicators):
        return {"unit": unit, "treatment": 0, "scenario": "untreated"}
    
    # Check for effect inquiry
    if any(phrase in query_l for phrase in effect_indicators):
        return {"unit": unit, "treatment": treatment, "scenario": "effect"}
    
    print("Could not determine scenario type")
    return None

# Load data
data = pd.read_csv('synthetic_causal_data_with_age.csv')
data['confounder_normalized'] = (
    (data['confounder'] - data['confounder'].mean()) / data['confounder'].std()
)

# Set up Hugging Face pipeline for LangChain with a text2text model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100,
    do_sample=False,
    num_beams=5,
    temperature=0.1
)
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"device": "cpu"})

# Define an enhanced prompt template with more natural language variations
template = """Return a JSON object in this exact format: {"unit": int, "treatment": int or null, "scenario": "treated" or "untreated" or "effect"}. No extra text or explanations. Parse queries about patients, units, or cases. Examples:
- "What if patient 17 takes medicine 1?" -> {"unit": 17, "treatment": 1, "scenario": "treated"}
- "What happens if unit 43 skips medication?" -> {"unit": 43, "treatment": 0, "scenario": "untreated"}
- "Show me the effect for patient #10" -> {"unit": 10, "treatment": null, "scenario": "effect"}
- "Will case 25 improve with treatment?" -> {"unit": 25, "treatment": 1, "scenario": "treated"}
- "Predict outcome for subject 8 without medicine" -> {"unit": 8, "treatment": 0, "scenario": "untreated"}
- "What's the impact on individual 33?" -> {"unit": 33, "treatment": null, "scenario": "effect"}"""
prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)
llm_chain = RunnableSequence(prompt_template | llm)

# Generate explanation function in child-friendly mode for all queries
def generate_explanation(unit, actual_outcome, counterfactual_outcome, confidence, query, rl_treatment, parsed):
    diff = actual_outcome - counterfactual_outcome
    action = "taking their medicine" if rl_treatment == 1 else "skipping their medicine"
    health_change = abs(int(diff))  # Simplify to a whole number for kids
    
    if diff > 0:
        return (f"Imagine a patient like unit {unit}! If they {parsed['scenario'].replace('untreated', 'skipped their medicine').replace('treated', 'took their medicine')} "
                f"for a day, they might feel a little worse because their health score could drop by {health_change} points, "
                f"from {int(actual_outcome)} to {int(counterfactual_outcome)}. Our smart helper, who learned from lots of patients, "
                f"thinks {action} is the best choice with {confidence*100:.0f}% confidence!")
    elif diff < 0:
        return (f"Imagine a patient like unit {unit}! If they {parsed['scenario'].replace('untreated', 'skipped their medicine').replace('treated', 'took their medicine')} "
                f"for a day, they might feel a bit better because their health score could jump by {health_change} points, "
                f"from {int(actual_outcome)} to {int(counterfactual_outcome)}. Our smart helper, who learned from lots of patients, "
                f"thinks {action} is the best choice with {confidence*100:.0f}% confidence!")
    else:
        return (f"Imagine a patient like unit {unit}! If they {parsed['scenario'].replace('untreated', 'skipped their medicine').replace('treated', 'took their medicine')} "
                f"for a day, they’d probably feel the same because their health score stays around {int(actual_outcome)}. "
                f"Our smart helper, who learned from lots of patients, thinks {action} is the best choice with {confidence*100:.0f}% confidence!")

# Fit linear model for counterfactual baseline and compute confidence
X = data[['treatment', 'confounder']]
y = data['outcome']
model_lr = LinearRegression()
model_lr.fit(X, y)
alpha = model_lr.intercept_
beta = model_lr.coef_[0]
gamma = model_lr.coef_[1]
env = CausalEnv(data, alpha, beta, gamma)
agent = QLearningAgent(env)

# Train (run once or load pre-trained q_table)
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, _, done, _, _ = env.step(action)
        state = next_state
optimal_actions = np.argmax(agent.q_table, axis=2)

# Function to get RL treatment
def get_rl_treatment(unit_idx, confounder):
    confounder_bin = agent.discretize_confounder(confounder)
    return optimal_actions[unit_idx, confounder_bin]

# Calculate initial confidence based on R²
r_squared = model_lr.score(X, y)
confidence = min(0.99, max(0.5, r_squared + 0.1))
print(f"Model R²: {r_squared:.4f}, Derived Confidence: {confidence*100:.0f}%")

# Interactive query loop with robust error handling
    
if __name__ == "__main__":
# Interactive query loop with robust error handling
  while True:
    try:
        query = input("Enter a query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        # Try fallback parser first (it's more reliable)
        parsed = fallback_parse(query)
        
        # If fallback fails, try LLM
        if parsed is None:
            try:
                print("Trying LLM parsing...")
                response = llm_chain.invoke({"query": query})
                if isinstance(response, dict) and 'text' in response:
                    response_text = response['text'].strip()
                else:
                    response_text = str(response).strip()
                
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        # Validate required fields
                        if not all(k in parsed for k in ['unit', 'treatment', 'scenario']):
                            print("LLM response missing required fields")
                            parsed = None
                    except json.JSONDecodeError:
                        print("Could not parse LLM JSON response")
                        parsed = None
            except Exception as e:
                print(f"LLM parsing failed: {str(e)}")
                parsed = None
        
        # If we still don't have a valid parse, show helpful message
        if parsed is None:
            print("\nCould not understand query. Please use format like:")
            print("- 'What if unit 5 gets treatment 1?'")
            print("- 'What happens if unit 3 skips treatment?'")
            print("- 'What is the effect on unit 7?'\n")
            continue
            
        print(f"Successfully parsed query: {parsed}")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        continue

    # Process the parsed query result
    if parsed and 'scenario' in parsed:
        if parsed['scenario'] in ["treated", "untreated"]:
            rl_treatment = parsed.get('treatment', get_rl_treatment(parsed['unit'], data.iloc[parsed['unit']]['confounder_normalized']))
            counterfactual_outcome = compute_counterfactual(parsed['unit'], rl_treatment)
            actual_outcome = data.iloc[parsed['unit']]['outcome']
            explanation = generate_explanation(parsed['unit'], actual_outcome, counterfactual_outcome, confidence, query, rl_treatment)
            print("Explanation:", explanation)
        else:
            ate = 2.0
            explanation = (f"Imagine a patient like unit {parsed['unit']}! If we check how medicine helps, "
                          f"our smart helper thinks it could make a difference of {int(ate)} health points "
                          f"with {confidence*100:.0f}% confidence!")
            print("Explanation:", explanation)

        # Log results
        with open('query_results.txt', 'a') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Parsed: {parsed}\n")
            f.write(f"Explanation: {explanation}\n\n")
    
    if parsed and 'scenario' in parsed:
        if parsed['scenario'] in ["treated", "untreated"]:
            rl_treatment = parsed.get('treatment', get_rl_treatment(parsed['unit'], data.iloc[parsed['unit']]['confounder_normalized']))
            counterfactual_outcome = compute_counterfactual(parsed['unit'], rl_treatment)
            actual_outcome = data.iloc[parsed['unit']]['outcome']
            explanation = generate_explanation(parsed['unit'], actual_outcome, counterfactual_outcome, confidence, query, rl_treatment)
            print("Explanation:", explanation)
        else:
            ate = 2.0
            explanation = (f"Imagine a patient like unit {parsed['unit']}! If we check how medicine helps, "
                          f"our smart helper thinks it could make a difference of {int(ate)} health points "
                          f"with {confidence*100:.0f}% confidence!")
            print("Explanation:", explanation)

    with open('query_results.txt', 'a') as f:
        f.write(f"Query: {query}\n")
        if 'parsed' in locals() and parsed and 'scenario' in parsed:
            f.write(f"Parsed: {parsed}\n")
            if parsed['scenario'] in ["treated", "untreated"]:
                rl_treatment = parsed.get('treatment', get_rl_treatment(parsed['unit'], data.iloc[parsed['unit']]['confounder_normalized']))
                counterfactual_outcome = compute_counterfactual(parsed['unit'], rl_treatment)
                actual_outcome = data.iloc[parsed['unit']]['outcome']
                explanation = generate_explanation(parsed['unit'], actual_outcome, counterfactual_outcome, confidence, query, rl_treatment)
                f.write(f"Explanation: {explanation}\n")
            else:
                ate = 2.0
                explanation = (f"Imagine a patient like unit {parsed['unit']}! If we check how medicine helps, "
                              f"our smart helper thinks it could make a difference of {int(ate)} health points "
                              f"with {confidence*100:.0f}% confidence!")
                f.write(f"Explanation: {explanation}\n")
        else:
            f.write(f"Parsing failed: {str(e) if 'e' in locals() else 'Unknown error'}\n")
        f.write("\n")


# Note: Move to Colab for this due to memory constraints
# from transformers import Trainer, TrainingArguments
# train_data = [...]  # 50 query-answer pairs
# tokenized_data = tokenizer(train_data['query'], padding=True, truncation=True, max_length=128)
# # Define Trainer and train (see Hugging Face docs for details)