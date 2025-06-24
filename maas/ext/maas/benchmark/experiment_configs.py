from typing import Dict, List

class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators

EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Programmer", "SelfRefine", "EarlyStop"],
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Programmer", "SelfRefine", "EarlyStop"],
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        # operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Test", "SelfRefine", "EarlyStop"],
        operators=["Generate"],
    ),
    "VeriThoughts": ExperimentConfig(
        dataset="VeriThoughts",
        question_type="code",
        # operators=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Test", "SelfRefine", "EarlyStop"],
        operators=["Generate"],
    ),
}
