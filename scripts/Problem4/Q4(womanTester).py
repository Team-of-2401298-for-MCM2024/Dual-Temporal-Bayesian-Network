from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import DiscreteFactor
import numpy as np

# Load your model
model = BayesianNetwork.load("WomanModel.bif")
inference = VariableElimination(model)

# Function to perform query
def query_with_evidence(evidence):
    result = inference.query(variables=['residual_effect_1'], evidence=evidence)
    return result

# Function to calculate entropy of a probability distribution
def calculate_entropy(distribution):
    if isinstance(distribution, DiscreteFactor):
        probabilities = distribution.values
    else:
        probabilities = distribution
    # Ensure probabilities are normalized
    probabilities = probabilities / np.sum(probabilities)
    entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # Adding epsilon to avoid log(0)
    return entropy

# Evidence and their possible values
evidence_values = {
    "double_fault_1": ["2", "1", "0"],
    "score_advantage_1": ["2", "1", "0"],
    "unf_err_1": ["2", "1", "0"],
    "break_pt_won_1": ["2", "1", "0"],
    "net_pt_won_1": ["2", "1", "0"],
    "ace_1": ["2", "1", "0"],
    "perception_of_control_1": ["1", "0"],
    "self_efficacy_1": ["1", "0"]
}

# Iterate over conditions and compute both the result and its entropy
for evidence, values in evidence_values.items():
    for value in values:
        result = query_with_evidence({evidence: value})
        entropy = calculate_entropy(result)
        print(f"Result for {evidence}={value}:\n{result}")
        print(f"Entropy: {entropy}\n")
