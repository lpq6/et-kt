"""Portable local training, with the server fold/seed and data contracts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    TrainValidationDataset,
    FixedOrderBatchSampler,
    collate_ordered_segments,
)
from .metrics import binary_metrics, exceeds_floor, selection_key
from .model import A2G
from .provenance import sha256, write_json, source_hashes
from .protocol import validate_training_protocol
from .randomness import A2GDropoutSubstreamAlignment
from .uid import load_uid_segment_metadata_csv, UIDSegmentDataset


ABLATIONS = {
    "no_umk_ssm": "use_umk_ssm",
    "no_umk_attn": "use_umk_attn",
    "no_umk_rwce": "use_umk_rwce",
    "no_transfer_residual": "use_transfer_residual",
    "no_causal_transfer_gate": "use_causal_transfer_gate",
    "no_aligned_history_statistics": "use_aligned_history_statistics",
    "no_attention": "use_attention_branch",
    "no_concept_graph": "use_concept_graph",
    "no_evidence": "use_evidence_branch",
    "no_factorized_input": "use_factorized_input",
    "no_ffn_gate": "use_ffn_gate",
    "no_history_pace": "use_history_pace",
    "no_item_attempt_stage": "use_item_attempt_stage",
    "no_input_memory": "use_input_memory",
    "no_multimode_residual": "use_multimode_residual",
    "no_prequential_newton": "use_prequential_newton",
    "no_ssm": "use_ssm_branch",
    "no_split_boundary": "use_split_boundary",
    "no_rwce": "use_recency_weighted_concept_evidence",
    "no_evidence_equivalent_residual": "use_evidence_equivalent_residual",
    "no_memory_readout": "use_memory_readout",
    "no_shared_embeddings": "use_shared_embeddings",
}


# Block-level ablation groups: group the 22 fine-grained switches into 5
# scientific blocks that follow the forward() data-flow pipeline.
# A "no_B*" variant disables every switch inside one block at once.
BLOCKS = {
    "B1_input_evidence": [
        "use_evidence_branch",
        "use_recency_weighted_concept_evidence",
        "use_evidence_equivalent_residual",
        "use_factorized_input",
        "use_history_pace",
        "use_shared_embeddings",
    ],
    "B2_state_memory": [
        "use_ssm_branch",
        "use_split_boundary",
        "use_input_memory",
        "use_multimode_residual",
    ],
    "B3_attention_fusion": [
        "use_attention_branch",
        "use_ffn_gate",
        "use_memory_readout",
    ],
    "B4_concept_routing": [
        "use_concept_graph",
        "use_transfer_residual",
        "use_causal_transfer_gate",
    ],
    "B5_calibrated_readout": [
        "use_prequential_newton",
        "use_item_attempt_stage",
    ],
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["dataset"] != "assist2017" or config["seed"] != 42:
        raise ValueError("the local protocol is Assist2017, seed42 only")
    if config["train_folds"] != [1, 2, 3, 4] or config["validation_folds"] != [0]:
        raise ValueError("fold protocol changed")
    if config["test_access"] is not False:
        raise ValueError("test access is not permitted")
    if config["data"]["maxlen"] != config["model"]["seq_len"]:
        raise ValueError("sequence width mismatch")
    validate_training_protocol(config)
    return config


def load_data(config, directory):
    directory = Path(directory)
    csv = directory / "train_valid_sequences.csv"
    mapping = directory / "training_fold_id_maps.json"
    binding = config["data_binding"]
    for file, expected in [
        (csv, binding["csv_sha256"]),
        (mapping, binding["mapping_sha256"]),
    ]:
        if not file.is_file():
            raise FileNotFoundError(f"Missing frozen data: {file}")
        if sha256(file) != expected:
            raise ValueError(f"Frozen data hash mismatch: {file}")
    bundles, uid_sets, counts = {}, {}, {}
    for split in ("train", "validation"):
        folds = config[split + "_folds"]
        base = TrainValidationDataset(csv, config["data"], folds)
        metadata = load_uid_segment_metadata_csv(
            csv,
            dataset_id=config["dataset"],
            split=split,
            allowed_folds=folds,
            sequence_width=config["data"]["maxlen"],
            expected_dataset_length=len(base),
        )
        bundles[split] = {
            "base": base,
            "metadata": metadata,
            "uid": UIDSegmentDataset(base, metadata),
        }
        uid_sets[split] = {entry.learner_uid for entry in metadata}
        counts[split] = {
            "segments": len(base),
            "learners": len(uid_sets[split]),
            "scored_interactions": base.scored_interactions,
        }
    if uid_sets["train"] & uid_sets["validation"]:
        raise ValueError("learner overlap between train and validation")
    if counts != binding["counts"]:
        raise ValueError(f"frozen population changed: {counts}")
    return bundles, {
        "csv_sha256": sha256(csv),
        "mapping_sha256": sha256(mapping),
        "counts": counts,
        "train_validation_uid_overlap": 0,
        "test_access": False,
    }


@torch.no_grad()
def initialize_prior(model, base):
    item_count = torch.zeros_like(model.item_prior.weight[:, 0], device="cpu")
    item_correct = torch.zeros_like(item_count)
    concept_count = torch.zeros_like(model.concept_prior.weight[:, 0], device="cpu")
    concept_correct = torch.zeros_like(concept_count)
    for batch in DataLoader(base, batch_size=64, shuffle=False):
        mask = batch["smasks"].bool()
        response = batch["shft_rseqs"][mask].float()
        concept = batch["shft_cseqs"][mask].long()
        concept_count.scatter_add_(0, concept, torch.ones_like(response))
        concept_correct.scatter_add_(0, concept, response)
        if batch["shft_qseqs"].numel():
            item = batch["shft_qseqs"][mask].long()
            item_count.scatter_add_(0, item, torch.ones_like(response))
            item_correct.scatter_add_(0, item, response)
    smoothing, scale = 24.0, 0.18
    global_rate = (concept_correct.sum() + smoothing) / (
        concept_count.sum() + 2 * smoothing
    )
    center = torch.logit(global_rate.clamp(1e-4, 1 - 1e-4))
    logits = []
    for correct, count in [
        (item_correct, item_count),
        (concept_correct, concept_count),
    ]:
        rate = (correct + smoothing * global_rate) / (count + smoothing)
        logits.append(scale * (torch.logit(rate.clamp(1e-4, 1 - 1e-4)) - center))
    model.set_prior_logits(*logits)
    model.set_prior_support(item_count, concept_count)
    return {
        "source": "training_folds_only",
        "selected_interactions": int(concept_count.sum()),
        "global_rate": float(global_rate),
        "smoothing": smoothing,
        "scale": scale,
    }


def loader(bundle, size, epoch=0, shuffle=False):
    sampler = FixedOrderBatchSampler(
        bundle["metadata"], size, seed=42, epoch=epoch, shuffle=shuffle
    )
    return DataLoader(
        bundle["uid"],
        batch_sampler=sampler,
        collate_fn=collate_ordered_segments,
        num_workers=0,
    )


def forward_aligned(model, batch, alignment, epoch, cursor, training=False):
    alignment.begin_step(epoch=epoch, cursor=cursor)
    try:
        output = model(batch, train=training)
        alignment.end_step()
        return output[0] if training else output
    except BaseException:
        if alignment._active:
            alignment.abort_step()
        raise


@torch.no_grad()
def evaluate(model, bundle, alignment, device, batch_size, epoch=0, limit=None):
    model.eval()
    labels, probabilities, learners, rows, positions = [], [], [], [], []
    for cursor, batch in enumerate(loader(bundle, batch_size)):
        if limit is not None and cursor >= limit:
            break
        dcur = {key: value.to(device) for key, value in batch["dcur"].items()}
        prediction = forward_aligned(model, dcur, alignment, epoch, 1000000 + cursor)
        mask = dcur["smasks"].bool()
        labels.append(dcur["shft_rseqs"][mask].cpu().numpy().astype(np.int8))
        probabilities.append(prediction[:, 1:][mask].cpu().numpy().astype(np.float32))
        for index, row_mask in enumerate(mask.cpu().numpy()):
            count = int(row_mask.sum())
            metadata = bundle["metadata"][batch["state_meta"]["dataset_index"][index]]
            learners.extend([metadata.learner_uid] * count)
            rows.extend([metadata.csv_row_index] * count)
            positions.extend((np.flatnonzero(row_mask) + 1).tolist())
    arrays = {
        "label": np.concatenate(labels),
        "probability": np.concatenate(probabilities),
        "learner_uid": np.asarray(learners),
        "csv_row_index": np.asarray(rows, dtype=np.int64),
        "position": np.asarray(positions, dtype=np.int16),
    }
    return binary_metrics(arrays["label"], arrays["probability"]), arrays


def make_model(config, variant):
    architecture = config.get("architecture", "v33")
    if variant == "no_aligned_history_statistics" and architecture != "v44_aligned_history":
        raise ValueError(f"{variant} requires the v44_aligned_history architecture")
    if (
        variant in {"no_input_memory", "no_multimode_residual"}
        and architecture not in {"v43", "v44_aligned_history", "v45_umk"}
    ):
        raise ValueError(f"{variant} requires the v43 architecture")
    kwargs = dict(config["model"])
    block_variants = {f"no_{b}": b for b in BLOCKS}
    matched_umk_control = architecture == "v45_umk" and variant in {
        "no_umk_ssm", "no_umk_attn", "no_umk_rwce",
    }
    matched_transfer_control = (
        architecture == "v47_transfer" and variant == "no_transfer_residual"
    )
    matched_causal_transfer_control = (
        architecture == "v48_causal_transfer"
        and variant == "no_causal_transfer_gate"
    )
    if variant in block_variants:
        for flag in BLOCKS[block_variants[variant]]:
            kwargs[flag] = 0
    elif variant != "full" and not (
        matched_umk_control
        or matched_transfer_control
        or matched_causal_transfer_control
    ):
        kwargs[ABLATIONS[variant]] = 0
    if architecture == "v45_umk":
        from .umk import A2GUMK

        model_class = A2GUMK
    elif architecture == "v47_transfer":
        from .transfer import A2GTransfer

        model_class = A2GTransfer
    elif architecture == "v48_causal_transfer":
        from .transfer_gate import A2GCausalTransferGate

        model_class = A2GCausalTransferGate
    elif architecture == "v44_aligned_history":
        from .aligned_history import A2GAlignedHistory

        model_class = A2GAlignedHistory
    elif architecture in ("v43", "v46"):
        from .candidate import A2GModal

        model_class = A2GModal
    elif config.get("architecture", "v33") == "v33":
        model_class = A2G
    else:
        raise ValueError("unknown architecture")
    model = model_class(config["data"]["num_c"], config["data"]["num_q"], **kwargs)
    if matched_umk_control:
        # Keep Full's complete random initial state; disabled scalars receive no
        # gradient. Plain all-off construction still has the legacy state dict.
        if variant == "no_umk_ssm":
            model.ssm.use_umk_ssm = False
        else:
            setattr(model, ABLATIONS[variant], False)
    elif matched_transfer_control:
        # Keep the Full initial state and scalar parameter for a matched
        # retrained control; disabling the path removes its forward/gradient
        # contribution without changing unrelated initialization.
        model.use_transfer_residual = False
    elif matched_causal_transfer_control:
        # Keep the Full initial state and scalar parameter for a matched
        # retrained control.  Disabling the path removes only its forward
        # contribution and leaves all shared tensors unchanged.
        model.use_causal_transfer_gate = False
    return model


def run(args, config, bundles, data_report):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    smoke = args.command == "smoke"
    batch_size = args.smoke_batch_size if smoke else config["training"]["batch_size"]
    evaluation_batch = (
        args.smoke_batch_size if smoke else config["training"]["evaluation_batch_size"]
    )
    epochs = 1 if smoke else config["training"]["max_epochs"]
    seed_all(42)
    model = make_model(config, args.variant)
    prior = initialize_prior(model, bundles["train"]["base"])
    torch.save(model.state_dict(), out / "initial_state.pt")
    model.to(device)
    opt_name = config["training"].get("optimizer", "adam")
    opt_cls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    optimizer = opt_cls(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = None
    warmup_epochs = int(config["training"].get("warmup_epochs", 0))
    if warmup_epochs > 0:

        def _lr_lambda(epoch_idx):
            if epoch_idx < warmup_epochs:
                return (epoch_idx + 1) / warmup_epochs
            progress = (epoch_idx - warmup_epochs) / max(
                1, epochs - warmup_epochs
            )
            cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
            return 0.1 + 0.9 * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    metadata = {
        "config": config,
        "training_protocol": validate_training_protocol(config),
        "variant": args.variant,
        "mode": args.command,
        "data": data_report,
        "prior": prior,
        "source": source_hashes(Path(__file__).parent),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "initial_state_sha256": sha256(out / "initial_state.pt"),
        "pretrained_checkpoint": None,
        "optimizer_reused": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "effective_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "microbatching": False,
        "effective_evaluation_batch_size": evaluation_batch,
        "independent_training_process_pid": os.getpid(),
    }
    write_json(out / "run.json", metadata)
    alignment = A2GDropoutSubstreamAlignment(model, base_seed=42)
    seed_all(42)
    started, best, best_epoch, best_metrics = time.monotonic(), None, None, None
    checkpoint = out / "selected_model.pt"
    try:
        with (out / "epochs.jsonl").open("x", encoding="utf-8") as log:
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss, count, updates = 0.0, 0, 0
                for cursor, batch in enumerate(
                    loader(
                        bundles["train"],
                        batch_size,
                        epoch - 1,
                        shuffle=config["training"].get("shuffle", False),
                    )
                ):
                    if smoke and cursor >= args.smoke_steps:
                        break
                    dcur = {
                        key: value.to(device) for key, value in batch["dcur"].items()
                    }
                    optimizer.zero_grad(set_to_none=True)
                    prediction = forward_aligned(
                        model, dcur, alignment, epoch - 1, cursor, True
                    )
                    mask = dcur["smasks"].bool()
                    loss = F.binary_cross_entropy(
                        prediction[:, 1:][mask].double(),
                        dcur["shft_rseqs"][mask].double(),
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    updates += 1
                    n = int(mask.sum())
                    total_loss += float(loss.detach()) * n
                    count += n
                metrics, _ = evaluate(
                    model,
                    bundles["validation"],
                    alignment,
                    device,
                    evaluation_batch,
                    epoch,
                    args.smoke_steps if smoke else None,
                )
                key = selection_key(metrics)
                if best is None or key > best:
                    best, best_epoch, best_metrics = key, epoch, metrics
                    torch.save(model.state_dict(), checkpoint)
                row = {
                    "epoch": epoch,
                    "train_loss": total_loss / count,
                    "train_interactions": count,
                    "optimizer_steps": updates,
                    "validation": metrics,
                    "best_epoch": best_epoch,
                    "best_auc": best[0],
                    "duration_seconds": time.monotonic() - started,
                    "smoke_only": smoke,
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                if scheduler is not None:
                    scheduler.step()
                if epoch - best_epoch >= config["training"]["patience"]:
                    break
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True), strict=True
        )
        metrics, predictions = evaluate(
            model,
            bundles["validation"],
            alignment,
            device,
            evaluation_batch,
            100,
            args.smoke_steps if smoke else None,
        )
        if metrics != best_metrics:
            raise RuntimeError("selected checkpoint metrics did not reproduce exactly")
        np.savez_compressed(out / "validation_predictions.npz", **predictions)
        result = {
            "status": "smoke_only_not_performance"
            if smoke
            else "complete_local_independent_validation",
            "dataset": "assist2017",
            "seed": 42,
            "training_protocol": validate_training_protocol(config),
            "variant": args.variant,
            "metrics": metrics,
            "best_epoch": best_epoch,
            "epochs": epoch,
            "strict_floor": 0.8174,
            "point_gate_pass": False if smoke else exceeds_floor(metrics["auc"]),
            "independent_terminal_audit": False,
            "paper_goal_complete": False,
            "baseline_comparability": "legacy floors only; remapped baselines need matching retraining",
            "checkpoint_sha256": sha256(checkpoint),
            "predictions_sha256": sha256(out / "validation_predictions.npz"),
            "run_sha256": sha256(out / "run.json"),
            "epoch_log_sha256": sha256(out / "epochs.jsonl"),
            "test_access": False,
            "duration_seconds": time.monotonic() - started,
        }
        write_json(out / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        write_json(
            out / "incomplete.json",
            {
                "status": "incomplete",
                "error_type": type(error).__name__,
                "message": str(error),
                "performance_failure": False,
                "duration_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        if alignment._active:
            alignment.abort_step()
        alignment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit-data", "smoke", "train"])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/assist2017_v33.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/assist2017"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    block_choices = [f"no_{b}" for b in BLOCKS]
    parser.add_argument(
        "--variant", choices=["full", *ABLATIONS, *block_choices], default="full"
    )
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.smoke_steps < 1 or args.smoke_batch_size < 1:
        parser.error("smoke sizes must be positive")
    config = load_config(args.config)
    torch.set_num_threads(1)
    bundles, report = load_data(config, args.data_dir)
    if args.command == "audit-data":
        print(json.dumps(report, indent=2))
        if args.output:
            write_json(args.output, report)
        return
    if not args.output:
        parser.error("--output must name a new experiment directory")
    if args.command == "train" and args.variant != "full":
        protocol = config.get("training_protocol", "")
        if not protocol.endswith("_ablation_v1"):
            parser.error(
                "variant training requires an explicit frozen ablation protocol "
                "(training_protocol ending in _ablation_v1)"
            )
    run(args, config, bundles, report)


if __name__ == "__main__":
    main()
