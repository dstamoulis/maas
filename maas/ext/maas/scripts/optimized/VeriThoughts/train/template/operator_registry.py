from maas.ext.maas.scripts.optimized.VeriThoughts.train.template.operator import (
    Generate,
    GenerateCoT,
    MultiGenerateCoT,
    ScEnsemble,
    Test,
    SelfRefine,
    EarlyStop
)

operator_mapping = {
    "Generate": Generate,
    "GenerateCoT": GenerateCoT,
    "MultiGenerateCoT": MultiGenerateCoT,
    "ScEnsemble": ScEnsemble,
    "Test": Test,
    "SelfRefine": SelfRefine,
    "EarlyStop": EarlyStop,
}

operator_mapping = {
    "Generate": Generate
}

operator_names = list(operator_mapping.keys())
