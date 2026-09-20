"""Explicit training protocols; batch256 never aliases the batch64 reference."""

REFERENCE = "assist2017_fold0_seed42_batch64_v1"
DIRECT_256 = "assist2017_fold0_seed42_batch256_direct_v1"
# v46: shared-embedding architecture with upgraded protocol
# (AdamW wd=1e-4, seeded per-epoch shuffle, 3-epoch warmup + cosine decay).
V46 = "assist2017_fold0_seed42_batch64_v46_v1"
V47 = "assist2017_fold0_seed42_batch64_v47_v1"
V48 = "assist2017_fold0_seed42_batch64_v48_v1"
# Frozen ablation protocol: same hyper-parameters as V46, distinct identifier so
# variant (ablation) runs are never mistaken for full admissions.
ABLATION_V46 = "assist2017_fold0_seed42_batch64_v46_ablation_v1"


def validate_training_protocol(config):
    identifier = config.get("training_protocol", REFERENCE)
    batches = {
        REFERENCE: 64,
        DIRECT_256: 256,
        V46: 64,
        V47: 64,
        V48: 64,
        ABLATION_V46: 64,
    }
    if identifier not in batches:
        raise ValueError(f"unknown training protocol: {identifier}")
    if identifier in {V46, V47, V48, ABLATION_V46}:
        expected = {
            "batch_size": 64,
            "evaluation_batch_size": 128,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "max_epochs": 200,
            "patience": 20,
            "optimizer": "adamw",
            "shuffle": True,
            "warmup_epochs": 3,
        }
    else:
        expected = {
            "batch_size": batches[identifier],
            "evaluation_batch_size": 128,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "max_epochs": 200,
            "patience": 20,
        }
    training = config["training"]
    if set(training) != set(expected):
        raise ValueError(f"optimizer/batch/stopping settings differ from {identifier}")
    for key, value in expected.items():
        if type(training[key]) is bool and not isinstance(value, bool):
            raise ValueError(
                f"optimizer/batch/stopping settings differ from {identifier}"
            )
        if training[key] != value:
            raise ValueError(
                f"optimizer/batch/stopping settings differ from {identifier}"
            )
    return identifier
