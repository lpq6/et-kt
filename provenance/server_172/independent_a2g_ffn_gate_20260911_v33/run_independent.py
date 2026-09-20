"""Independent validation experiments with immutable inputs and learner keys."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
DATA_CONFIG = ROOT / "data_config.train_fold_only.json"
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))
PROTOCOL = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
from gpu_guard import check as check_gpu_resources
from gpu_guard import require_available
from launch_candidate import validate_protocol

def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def verify_manifest():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or digest(path) != expected:
            raise RuntimeError(f"package verification failed: {relative}")
    return record(ROOT / "manifest.json")


def verify_closed_predecessor():
    binding = PROTOCOL["closed_predecessor"]
    path = Path(binding["result_path"])
    if digest(path) != binding["result_sha256"]:
        raise RuntimeError("predecessor result binding changed")
    result = json.loads(path.read_text())
    if result["candidate_id"] != binding["candidate_id"] or result["status"] != binding["required_status"]:
        raise RuntimeError("predecessor was not the required completed candidate")
    if result["metrics"]["auc"] != binding["auc"]:
        raise RuntimeError("predecessor metric changed")
    if result["metrics"]["auc"] > PROTOCOL["reference_auc"]["assist2017"]:
        raise RuntimeError("predecessor passed; do not launch another screen")
    return record(path)


def verify_best_candidate(initial):
    binding = PROTOCOL["best_candidate"]
    result_path = Path(binding["result_path"])
    if digest(result_path) != binding["result_sha256"]:
        raise RuntimeError("best-candidate result binding changed")
    result = json.loads(result_path.read_text())
    if (
        result["candidate_id"] != binding["candidate_id"]
        or result["status"] != "complete_independent_random_init_validation"
        or result["metrics"]["auc"] != binding["auc"]
    ):
        raise RuntimeError("best-candidate identity or metric mismatch")
    initial_path = Path(binding["initial_state_path"])
    if digest(initial_path) != binding["initial_state_sha256"]:
        raise RuntimeError("best-candidate initialization binding changed")
    reference = torch.load(initial_path, map_location="cpu", weights_only=True)
    added = set(PROTOCOL["candidate_structure"]["new_state_tensors"])
    expected_names = {
        f"blocks.{block}.ffn.gate_{suffix}"
        for block in range(3) for suffix in ("weight", "bias")
    }
    if added != expected_names:
        raise RuntimeError("exactly six declared FFN-gate parameters are required")
    if (
        len(reference) != PROTOCOL["candidate_structure"]["expected_parent_state_tensors"]
        or set(initial) != set(reference) | added
        or not all(
        isinstance(value, torch.Tensor) and isinstance(initial[name], torch.Tensor)
        and initial[name].dtype == value.dtype and bool(torch.isfinite(value).all())
        and initial[name].shape == value.shape
        and torch.equal(initial[name], value) for name, value in reference.items()
        )
    ):
        raise RuntimeError("independent initialization differs from v32")
    width = reference["ssm.in_proj.weight"].shape[1]
    hidden = PROTOCOL["model_kwargs"]["d_ff"]
    if width < 1 or hidden < 1 or PROTOCOL["model_kwargs"]["n_blocks"] != 4:
        raise RuntimeError("FFN-gate requires positive original widths and four attention blocks")
    shapes = {
        f"blocks.{block}.ffn.gate_{suffix}": (hidden, width) if suffix == "weight" else (hidden,)
        for block in range(3) for suffix in ("weight", "bias")
    }
    if any(
        initial[name].shape != shapes[name]
        or initial[name].dtype != reference["ssm.in_proj.weight"].dtype
        or not bool(torch.isfinite(initial[name]).all())
        for name in added
    ):
        raise RuntimeError("FFN-gate parameters must be finite and have the declared shapes and dtype")
    if any(not bool(initial[name].eq(0).all()) for name in added):
        raise RuntimeError("FFN-gate parameters must all be zero initialized")
    expected = {
        f"blocks.{block}.ffn.gate_{suffix}": torch.zeros_like(reference[f"blocks.{block}.ffn.0.{suffix}"])
        for block in range(3) for suffix in ("weight", "bias")
    }
    if any(not torch.equal(initial[name], expected[name]) for name in added):
        raise RuntimeError("FFN-gate initialization differs from the original FFN zero geometry")
    return {
        "status": "pass",
        "result": record(result_path),
        "initial_state": record(initial_path),
        "shared_tensors": len(reference),
        "new_tensors": sorted(added),
        "all_common_tensors_identical": True,
        "all_new_parameters_zero_initialized": True,
        "new_initialization_consumes_rng": False,
        "new_initial_parameters_match_seed42": True,
        "new_initial_tensor_sha256": {
            name: hashlib.sha256(initial[name].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name in sorted(added)
        },
        "new_parameter_count": sum(initial[name].numel() for name in added),
        "artifact_used_for_initialization": False,
        "trained_checkpoint_loaded": False,
    }


def verify_dataset_initialization(initial, config, data, dataset, seed):
    validate_protocol(PROTOCOL, dataset, seed, "full")
    return verify_incumbent_initialization(initial), verify_best_candidate(initial)


def verify_dataset_order(dataset):
    validate_protocol(PROTOCOL, dataset, 42, "full")
    return {
        "completed_predecessors": [], "point_failures": 0, "order": ["assist2017"],
        "assist2017_first_gate": PROTOCOL["assist2017_first_gate"],
        "other_dataset_execution_authorized": False,
    }


def setup():
    from strict_sequence_data import (
        TrainValidationDataset, FixedOrderBatchSampler, collate_ordered_segments,
    )
    from a2g_single_trunk_candidate import A2GMambaKT
    from a2g_dropout_substream_alignment_20260807 import A2GDropoutSubstreamAlignment
    from a2g_uid_topological_loader_20260807 import (
        UIDSegmentDataset,
        load_uid_segment_metadata_csv,
    )
    return {
        "dataset": TrainValidationDataset,
        "model": A2GMambaKT,
        "align": A2GDropoutSubstreamAlignment,
        "sampler": FixedOrderBatchSampler,
        "uid_dataset": UIDSegmentDataset,
        "collate": collate_ordered_segments,
        "metadata": load_uid_segment_metadata_csv,
    }


def verify_incumbent_initialization(initial):
    binding = PROTOCOL["incumbent"]
    path = Path(binding["initial_state_path"])
    if digest(path) != binding["initial_state_sha256"]:
        raise RuntimeError("incumbent initial-state hash changed")
    expected = torch.load(path, map_location="cpu", weights_only=True)
    new_tensors = {
        "semantic_stream.0.weight",
        "semantic_stream.0.bias",
        "semantic_stream.1.weight",
        "semantic_stream.1.bias",
        "evidence_stream.0.weight",
        "evidence_stream.0.bias",
        "evidence_stream.1.weight",
        "evidence_stream.1.bias",
        "factorized_residual.weight",
        "factorized_residual.bias",
    } | set(PROTOCOL["candidate_structure"]["inherited_auxiliary_state_tensors"]) | set(PROTOCOL["candidate_structure"]["new_state_tensors"])
    if set(initial) - set(expected) != new_tensors:
        raise RuntimeError(
            "candidate changed more than the declared factorized input, attempt, history pace, Newton, concept graph and FFN gates"
        )
    if not set(expected).issubset(initial):
        raise RuntimeError("candidate removed incumbent tensors")
    if not all(
        initial[name].dtype == tensor.dtype and initial[name].shape == tensor.shape
        and torch.equal(initial[name], tensor) for name, tensor in expected.items()
    ):
        raise RuntimeError("candidate random initialization does not preserve the incumbent")
    if not bool(initial["factorized_residual.weight"].eq(0).all()):
        raise RuntimeError("factorized residual weight is not zero initialized")
    if not bool(initial["factorized_residual.bias"].eq(0).all()):
        raise RuntimeError("factorized residual bias is not zero initialized")
    return {
        "status": "pass",
        "shared_tensor_count": len(expected),
        "added_tensors": sorted(new_tensors),
        "source": record(path),
        "trained_checkpoint_loaded": False,
        "initialization_taken_from_artifact": False,
    }


def model_kwargs(dataset):
    kwargs = dict(PROTOCOL["model_kwargs"])
    kwargs["item_residual_dropout"] = PROTOCOL["item_dropout_override"].get(
        dataset, kwargs["item_residual_dropout"]
    )
    return kwargs


def dataset_config(dataset):
    if dataset != "assist2017" or PROTOCOL["datasets"] != ["assist2017"]:
        raise RuntimeError("only Assist2017 is admitted in this package")
    config = json.loads(DATA_CONFIG.read_text())[dataset]
    if config["maxlen"] != PROTOCOL["model_kwargs"]["seq_len"]:
        raise RuntimeError(f"sequence width changed for {dataset}")
    if config["folds"] != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"folds changed for {dataset}")
    csv_path = Path(config["dpath"]) / config["train_valid_file"]
    if csv_path.name != "train_valid_sequences.csv" or not csv_path.is_file():
        raise RuntimeError(f"unexpected data source for {dataset}")
    summary = json.loads((ROOT / "trainfold_data_summary.json").read_text())
    binding = summary["datasets"][dataset]
    mapping_path = csv_path.parent / "training_fold_id_maps.json"
    if digest(csv_path) != binding["transformed_sha256"] or digest(mapping_path) != binding["mapping_sha256"]:
        raise RuntimeError(f"train-fold data binding mismatch: {dataset}")
    for field, parameter in (("concepts", "num_c"), ("questions", "num_q")):
        expected = binding["known_vocabulary_sizes"].get(field)
        expected = expected + 2 if expected is not None else 0
        if config[parameter] != expected:
            raise RuntimeError(f"embedding capacity mismatch: {dataset}/{field}")
    return config, csv_path


def load_data(api, dataset):
    config, csv_path = dataset_config(dataset)
    bundles = {}
    uid_sets = {}
    for split, folds in (
        ("train", PROTOCOL["train_folds"]),
        ("validation", PROTOCOL["validation_folds"]),
    ):
        base = api["dataset"](csv_path, config, frozenset(folds))
        metadata = api["metadata"](
            csv_path,
            dataset_id=dataset,
            split=split,
            allowed_folds=frozenset(folds),
            sequence_width=config["maxlen"],
            expected_dataset_length=len(base),
        )
        bundles[split] = {
            "base": base,
            "meta": metadata,
            "uid": api["uid_dataset"](base, metadata),
        }
        uid_sets[split] = {entry.learner_uid for entry in metadata}
    overlap = uid_sets["train"] & uid_sets["validation"]
    if overlap:
        raise RuntimeError(f"learner overlap in {dataset}: {len(overlap)}")
    provenance = {
        "sequence_csv": record(csv_path),
        "data_config": record(DATA_CONFIG),
        "cache_files": [],
        "cache_policy": "CSV parsed on CPU without reading or writing pickle caches",
        "training_fold_id_maps": record(csv_path.parent / "training_fold_id_maps.json"),
        "vocabulary_summary": record(ROOT / "trainfold_data_summary.json"),
        "train_validation_uid_overlap": 0,
        "counts": {
            split: {
                "segments": len(bundle["meta"]), "learners": len(uid_sets[split]),
                "scored_interactions": bundle["base"].scored_interactions,
            }
            for split, bundle in bundles.items()
        },
        "uid_hashes": {
            split: hashlib.sha256(
                "\n".join(sorted(values)).encode("utf-8")
            ).hexdigest()
            for split, values in uid_sets.items()
        },
        "vocabulary_training_only_provenance": "hash_bound_training_folds_only",
    }
    return config, bundles, provenance


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


@torch.no_grad()
def initialize_prior(model, base):
    item_count = torch.zeros_like(model.item_prior.weight[:, 0], device="cpu")
    item_correct = torch.zeros_like(item_count)
    concept_count = torch.zeros_like(model.concept_prior.weight[:, 0], device="cpu")
    concept_correct = torch.zeros_like(concept_count)
    for batch in DataLoader(base, batch_size=64, shuffle=False):
        mask = batch["smasks"].cpu().bool()
        responses = batch["shft_rseqs"].cpu()[mask].float()
        concepts = batch["shft_cseqs"].cpu()[mask].long()
        if concepts.min() < 0 or concepts.max() >= concept_count.numel():
            raise ValueError("concept identifier outside configured vocabulary")
        concept_count.scatter_add_(0, concepts, torch.ones_like(responses))
        concept_correct.scatter_add_(0, concepts, responses)
        questions = batch.get("shft_qseqs")
        if questions is not None and questions.numel():
            selected = questions.cpu()[mask].long()
            if selected.min() < 0 or selected.max() >= item_count.numel():
                raise ValueError("item identifier outside configured vocabulary")
            item_count.scatter_add_(0, selected, torch.ones_like(responses))
            item_correct.scatter_add_(0, selected, responses)
    smoothing, scale = 24.0, 0.18
    global_rate = (concept_correct.sum() + smoothing) / (
        concept_count.sum() + 2 * smoothing
    )
    item_rate = (item_correct + smoothing * global_rate) / (item_count + smoothing)
    concept_rate = (concept_correct + smoothing * global_rate) / (
        concept_count + smoothing
    )
    center = torch.logit(global_rate.clamp(1e-4, 1 - 1e-4))
    model.set_prior_logits(
        scale * (torch.logit(item_rate.clamp(1e-4, 1 - 1e-4)) - center),
        scale * (torch.logit(concept_rate.clamp(1e-4, 1 - 1e-4)) - center),
    )
    model.set_prior_support(item_count, concept_count)
    return {
        "source": "training_folds_only",
        "selected_interactions": int(concept_count.sum()),
        "global_rate": float(global_rate),
        "smoothing": smoothing,
        "scale": scale,
    }


def loader(api, bundle, seed, batch_size, epoch):
    sampler = api["sampler"](bundle["meta"], batch_size, seed=seed, epoch=epoch)
    return DataLoader(
        bundle["uid"], batch_sampler=sampler, collate_fn=api["collate"], num_workers=0
    ), sampler


def scores(labels, probabilities):
    probability = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    target = labels.astype(np.float64)
    if not np.isfinite(probability).all() or np.unique(target).size != 2:
        raise ValueError("invalid predictions or labels")
    ece = 0.0
    bins = np.minimum((probability * 15).astype(int), 14)
    for index in range(15):
        selected = bins == index
        if selected.any():
            ece += selected.mean() * abs(target[selected].mean() - probability[selected].mean())
    return {
        "auc": float(roc_auc_score(target, probability)),
        "acc": float(np.mean((probability >= 0.5) == target)),
        "nll": float(np.mean(-target * np.log(probability) - (1 - target) * np.log1p(-probability))),
        "brier": float(np.mean((target - probability) ** 2)),
        "ece_15bin": float(ece),
        "interaction_count": int(target.size),
        "positive_count": int(target.sum()),
    }


def forward_aligned(align, epoch, cursor, callback):
    align.begin_step(epoch=epoch, cursor=cursor)
    try:
        value = callback()
        align.end_step()
        return value
    except BaseException:
        if align._active:
            align.abort_step()
        raise


@torch.no_grad()
def evaluate(api, bundle, model, seed, device, align, epoch, include_keys=False):
    model.eval()
    batches, sampler = loader(api, bundle, seed, PROTOCOL["training"]["evaluation_batch_size"], 0)
    labels, probabilities, learners, rows, positions = [], [], [], [], []
    for cursor, batch in enumerate(batches):
        dcur = {key: value.to(device) for key, value in batch["dcur"].items()}
        prediction = forward_aligned(
            align, epoch, 1000000 + cursor, lambda: model(dcur)
        )
        mask = dcur["smasks"].bool()
        labels.append(dcur["shft_rseqs"][mask].cpu().numpy().astype(np.int8))
        probabilities.append(prediction[:, 1:][mask].cpu().numpy().astype(np.float32))
        if include_keys:
            for index, row_mask in enumerate(mask.cpu().numpy()):
                count = int(row_mask.sum())
                meta = batch["state_meta"]
                learners.extend([str(meta["learner_uid"][index])] * count)
                dataset_index = int(meta["dataset_index"][index])
                rows.extend([bundle["meta"][dataset_index].csv_row_index] * count)
                positions.extend((np.flatnonzero(row_mask) + 1).tolist())
    arrays = {
        "label": np.concatenate(labels),
        "probability": np.concatenate(probabilities),
    }
    if include_keys:
        arrays.update(
            learner_uid=np.asarray(learners),
            csv_row_index=np.asarray(rows, dtype=np.int64),
            position=np.asarray(positions, dtype=np.int16),
        )
    return scores(arrays["label"], arrays["probability"]), arrays, sampler.state_dict()


def resource_watchdog(stop, output):
    while not stop.wait(2):
        report = check_gpu_resources()
        if report["allowed"]:
            continue
        try:
            write_new(output / "resource_stop.json", {
                "status": "interrupted_resource_policy",
                "own_pid": os.getpid(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "resource_report": report,
            })
        finally:
            os._exit(75)


def train_variant(api, dataset, config, data, seed, flags, initial, out, variant):
    validate_protocol(PROTOCOL, dataset, seed, variant)
    out.mkdir()
    seed_all(seed)
    device = torch.device("cuda:0")
    if variant != "full" or PROTOCOL["controls"] or PROTOCOL["execution_order"] != ["full"]:
        raise RuntimeError("this frozen package permits one candidate Full only")
    model_class = api["model"]
    model = model_class(config["num_c"], config["num_q"], **model_kwargs(dataset), **flags)
    model.load_state_dict(initial, strict=True)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=PROTOCOL["training"]["learning_rate"],
        weight_decay=PROTOCOL["training"]["weight_decay"],
    )
    align = api["align"](model, base_seed=seed)
    seed_all(seed)
    best_key, best_epoch, history = None, None, []
    checkpoint = out / "selected_model.pt"
    started = time.perf_counter()
    try:
        with (out / "epochs.jsonl").open("x", encoding="utf-8") as log:
            for epoch in range(1, PROTOCOL["training"]["max_epochs"] + 1):
                model.train()
                batches, sampler = loader(
                    api, data["train"], seed, PROTOCOL["training"]["batch_size"], epoch - 1
                )
                loss_total, interaction_count = 0.0, 0
                for cursor, batch in enumerate(batches):
                    dcur = {key: value.to(device) for key, value in batch["dcur"].items()}
                    optimizer.zero_grad(set_to_none=True)
                    prediction = forward_aligned(
                        align, epoch - 1, cursor, lambda: model(dcur, train=True)[0]
                    )
                    mask = dcur["smasks"].bool()
                    loss = functional.binary_cross_entropy(
                        prediction[:, 1:][mask].double(),
                        dcur["shft_rseqs"][mask].double(),
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite loss: {dataset}/{variant}/{epoch}")
                    loss.backward()
                    optimizer.step()
                    count = int(mask.sum())
                    loss_total += float(loss.detach()) * count
                    interaction_count += count
                metrics, _, _ = evaluate(
                    api, data["validation"], model, seed, device, align, epoch
                )
                key = (
                    metrics["auc"], -metrics["nll"], -metrics["brier"],
                    -metrics["ece_15bin"], -epoch,
                )
                improved = best_key is None or key > best_key
                if improved:
                    torch.save(model.state_dict(), checkpoint)
                    best_key, best_epoch = key, epoch
                row = {
                    "dataset": dataset, "seed": seed, "variant": variant,
                    "epoch": epoch, "best_epoch": best_epoch, "best_auc": best_key[0],
                    "validation": metrics, "checkpoint_updated": improved,
                    "train_loss": loss_total / interaction_count,
                    "schedule": sampler.state_dict(),
                }
                history.append(row)
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                if epoch - best_epoch >= PROTOCOL["training"]["patience"]:
                    break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        metrics, arrays, schedule = evaluate(
            api, data["validation"], model, seed, device, align, 100, include_keys=True
        )
        if metrics["auc"] != best_key[0]:
            raise RuntimeError("selected checkpoint evaluation did not reproduce")
        prediction_path = out / "validation_predictions.npz"
        np.savez_compressed(prediction_path, **arrays)
        result = {
            "status": "complete_independent_random_init_validation",
            "dataset": dataset, "seed": seed, "variant": variant,
            "flags": flags,
            "candidate_id": PROTOCOL["candidate_id"],
            "pretrained_checkpoint": None, "optimizer_reused": False,
            "initial_state": record(out.parent / "initial_state.pt"),
            "metrics": metrics, "best_epoch": best_epoch, "epochs": len(history),
            "checkpoint_selection": PROTOCOL["training"]["checkpoint_selection"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "checkpoint": record(checkpoint), "predictions": record(prediction_path),
            "epoch_log": record(out / "epochs.jsonl"),
            "validation_schedule": schedule,
            "duration_seconds": time.perf_counter() - started,
            "test_access": False, "window_test_access": False,
        }
        write_new(out / "result.json", result)
        return result
    finally:
        align.close()
        del model, optimizer
        torch.cuda.empty_cache()


def preflight():
    validate_protocol(PROTOCOL, "assist2017", 42, "full")
    from audit_contract import audit
    from audit_data_contract import audit as audit_data
    manifest = verify_manifest()
    api = setup()
    sources = {}
    for dataset in PROTOCOL["datasets"]:
        config, path = dataset_config(dataset)
        model = api["model"](config["num_c"], config["num_q"], **model_kwargs(dataset))
        sources[dataset] = {
            "data": record(path),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "kwargs": model_kwargs(dataset),
        }
        del model
    contract = audit("cpu")
    if contract["status"] != "pass":
        raise RuntimeError(f"behavioral contract failed: {contract}")
    data_contract = audit_data()
    if data_contract["status"] != "pass":
        raise RuntimeError("data/control behavioral contract failed")
    guard_contract = json.loads((ROOT / "gpu_guard_contract.json").read_text())
    if guard_contract["status"] != "pass":
        raise RuntimeError("GPU ownership contract failed")
    return {
        "status": "pass", "manifest": manifest, "datasets": sources,
        "synthetic_contract": contract,
        "data_control_contract": data_contract,
        "gpu_ownership_contract": guard_contract,
        "cuda_initialized": torch.cuda.is_initialized(),
        "test_access": False, "window_test_access": False,
    }


def audit_production_cpu(dataset, seed):
    validate_protocol(PROTOCOL, dataset, seed, "full")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU production audit requires an explicitly empty CUDA_VISIBLE_DEVICES")
    manifest = verify_manifest()
    predecessor = verify_closed_predecessor()
    api = setup()
    config, data, provenance = load_data(api, dataset)
    seed_all(seed)
    model = api["model"](config["num_c"], config["num_q"], **model_kwargs(dataset))
    prior = initialize_prior(model, data["train"]["base"])
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    incumbent, best = verify_dataset_initialization(initial, config, data, dataset, seed)
    if torch.cuda.is_initialized() or any(parameter.is_cuda for parameter in model.parameters()):
        raise RuntimeError("CPU audit unexpectedly initialized CUDA")
    structure = PROTOCOL["candidate_structure"]
    gate_parameters = sum(initial[name].numel() for name in structure["new_state_tensors"])
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != structure["expected_assist2017_parameters"]
        or len(initial) != structure["expected_total_state_tensors"]
        or gate_parameters != structure["new_parameters"]
        or not all(type(block["attn"]) is torch.nn.MultiheadAttention for block in model.blocks)
        or model.n_question != 96
        or model.concept_graph.edge_weights(model.hist_concept_emb.weight[:model.n_question]).shape != (94, 94)
        or sum("ffn" in block for block in model.blocks) != 3
        or not all(block["ffn"].enabled for block in model.blocks if "ffn" in block)
    ):
        raise RuntimeError("production parameter count, tensor count or inherited attention type changed")
    unit_counts = {}
    with Path(provenance["sequence_csv"]["path"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            repeat = [int(value) for value in row["is_repeat"].split(",")]
            masks = [int(value) for value in row["selectmasks"].split(",")]
            if len(repeat) != 200 or len(masks) != 200:
                raise RuntimeError("attempt-unit sequence width changed")
            if any(flag != 0 for flag, mask in zip(repeat, masks) if mask == 1):
                raise RuntimeError("multi-concept repeated rows are not admitted by the ordinal attempt contract")
            unit_counts[row["fold"]] = unit_counts.get(row["fold"], 0) + masks.count(1)
    return {
        "status": "pass", "manifest": manifest, "closed_predecessor": predecessor,
        "dataset": dataset, "seed": seed, "data": provenance, "prior": prior,
        "incumbent_initialization": incumbent, "v32_initialization": best,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "gpu_used": False, "cuda_initialized": False,
        "trained_checkpoint_loaded": False, "optimizer_steps": 0,
        "ffn_gate_structure_diagnostic": {
            "total_state_tensors": len(initial),
            "new_parameters": gate_parameters,
            "gate_shapes": {name: list(initial[name].shape) for name in structure["new_state_tensors"]},
            "all_new_parameters_zero_initialized": all(bool(initial[name].eq(0).all()) for name in structure["new_state_tensors"]),
            "new_initial_parameters_match_seed42": best["new_initial_parameters_match_seed42"],
            "retained_graph_rank": model.concept_graph.rank,
            "declared_capacity": model.n_question,
            "known_graph_nodes": model.n_question - 2,
            "parent_spare_embedding_row_excluded": True,
            "new_path_nested_in_attention": True,
            "new_path_before_ssm": False,
            "gated_blocks": [index for index, block in enumerate(model.blocks) if "ffn" in block],
            "retained_newton_projection_rank": model.prequential_newton.rank,
            "retained_fisher_state_width": model.prequential_newton.rank + 1,
            "sequence_width": PROTOCOL["model_kwargs"]["seq_len"],
            "attention_layers": len(model.blocks),
            "all_original_attention_modules_preserved": all(type(block["attn"]) is torch.nn.MultiheadAttention for block in model.blocks),
            "active_initial_function_equals_v32": True,
            "retained_newton_output_zero_initialized": bool(model.prequential_newton.output_scale.eq(0)),
            "retained_history_pace_zero_initialized": all(bool(p.eq(0).all()) for p in model.history_pace.parameters()),
            "retained_attempt_output_zero_initialized": bool(model.attempt_readout.output.weight.eq(0).all()),
            "retained_graph_output_zero_initialized": bool(model.concept_graph.output.weight.eq(0).all()),
        },
        "inherited_history_time_training_profile": record(ROOT / "review/history_time_train_profile.json"),
        "attempt_unit_contract": {
            "all_valid_is_repeat_flags_zero": True,
            "valid_interactions_per_fold": unit_counts,
            "test_access": False,
        },
        "training_started": False, "test_access": False,
        "paper_goal_complete": False,
    }


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--audit-production-cpu", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--dataset", choices=PROTOCOL["datasets"])
    parser.add_argument("--seed", type=int, choices=PROTOCOL["seeds"], default=42)
    parser.add_argument("--variant", choices=PROTOCOL["execution_order"])
    args = parser.parse_args()
    validate_protocol(PROTOCOL, args.dataset or "assist2017", args.seed, args.variant or "full")
    torch.set_num_threads(1)
    if args.preflight_only:
        report = preflight()
        write_new(ROOT / "remote_preflight.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    if args.audit_production_cpu:
        if args.dataset is None:
            parser.error("--dataset is required for a production audit")
        report = audit_production_cpu(args.dataset, args.seed)
        write_new(ROOT / f"production_cpu_audit_{args.dataset}.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    if args.dataset is None:
        parser.error("--dataset is required for execution")
    manifest = verify_manifest()
    predecessor = verify_closed_predecessor()
    previous = json.loads((ROOT / "remote_preflight.json").read_text())
    if previous["status"] != "pass" or previous["manifest"]["sha256"] != manifest["sha256"]:
        raise RuntimeError("matching remote preflight is required")
    cpu_audit = json.loads((ROOT / f"production_cpu_audit_{args.dataset}.json").read_text())
    if (
        cpu_audit["status"] != "pass"
        or cpu_audit["manifest"]["sha256"] != manifest["sha256"]
        or (cpu_audit["dataset"], cpu_audit["seed"]) != (args.dataset, args.seed)
        or cpu_audit["cuda_initialized"]
    ):
        raise RuntimeError("matching dataset-specific CPU audit is required")
    dataset_progress = verify_dataset_order(args.dataset)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("physical GPU1 only")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic cuBLAS configuration missing")
    addresses = subprocess.check_output(["hostname", "-I"], text=True, timeout=10).split()
    if "172.25.114.0" not in addresses or os.getuid() != 1007:
        raise RuntimeError("execution is restricted to lpq on 172.25.114.0")
    gpu_uuid = subprocess.check_output([
        "nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True, timeout=10).strip()
    if gpu_uuid != "GPU-e1065296-99f9-bf6e-7898-d6755ca37a9d":
        raise RuntimeError("physical GPU1 identity changed")
    initial_resources = require_available()
    import fcntl
    lock = open("/home/lpq/a2g_mambakt/locks/GLOBAL_GPU_SERIAL_172_CONCEPT_LOCAL.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require_available()
    output = ROOT / "artifacts" / args.dataset / f"seed{args.seed}"
    if output.exists() and args.variant is None:
        raise RuntimeError("existing cells require an explicit unstarted variant")
    output.mkdir(parents=True, exist_ok=True)
    if args.variant and (output / args.variant).exists():
        raise RuntimeError("variant already started; never overwrite or retry silently")
    if (output / "resource_stop.json").exists():
        raise RuntimeError("interrupted package requires explicit audit, not automatic retry")
    stop = threading.Event()
    threading.Thread(target=resource_watchdog, args=(stop, output), daemon=True).start()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA device is required")
        if torch.cuda.mem_get_info()[0] < 16 * 1024 ** 3:
            raise RuntimeError("insufficient GPU1 memory")
        from audit_contract import audit
        seed_all(args.seed)
        contract = audit("cuda:0")
        if not (output / "cuda_contract.json").exists():
            write_new(output / "cuda_contract.json", contract)
        if contract["status"] != "pass":
            raise RuntimeError("CUDA behavioral contract failed")
        api = setup()
        config, data, provenance = load_data(api, args.dataset)
        seed_all(args.seed)
        model = api["model"](config["num_c"], config["num_q"], **model_kwargs(args.dataset))
        prior = initialize_prior(model, data["train"]["base"])
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        incumbent_initialization, best_initialization = verify_dataset_initialization(
            initial, config, data, args.dataset, args.seed,
        )
        del model
        for filename, state in (("initial_state.pt", initial),):
            state_path = output / filename
            if state_path.exists():
                persisted = torch.load(state_path, map_location="cpu", weights_only=True)
                if set(state) != set(persisted) or not all(torch.equal(state[name], persisted[name]) for name in state):
                    raise RuntimeError(f"initialization changed between independent cells: {filename}")
            else:
                torch.save(state, state_path)
        imported = {
            name: record(Path(module.__file__))
            for name, module in sys.modules.items()
            if (name.startswith("pykt.") or name.startswith("a2g_") or name.startswith("work.a2g_") or name in {"factorized_input_candidate", "attempt_stage_candidate", "history_pace_candidate", "prequential_newton_candidate", "concept_graph_candidate", "ffn_gate_candidate", "gpu_guard", "a2g_single_trunk_candidate", "launch_candidate"})
            and getattr(module, "__file__", None)
            and Path(module.__file__).is_file()
        }
        provenance_value = {
            "manifest": manifest, "data": provenance, "prior": prior,
            "closed_predecessor_result": predecessor,
            "dataset": args.dataset, "seed": args.seed,
            "kwargs": model_kwargs(args.dataset), "initial_state": record(output / "initial_state.pt"),
            "incumbent_initialization_parity": incumbent_initialization,
            "v32_initialization_parity": best_initialization,
            "initial_gpu_resource_report": initial_resources,
            "dataset_progress_before_start": dataset_progress,
            "production_cpu_audit": record(ROOT / f"production_cpu_audit_{args.dataset}.json"),
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "imports": imported,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "test_access": False, "window_test_access": False,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        provenance_path = output / "provenance.json"
        if provenance_path.exists():
            old = json.loads(provenance_path.read_text())
            for field in ("manifest", "data", "prior", "kwargs", "initial_state", "imports", "torch", "numpy", "cuda"):
                if old[field] != provenance_value[field]:
                    raise RuntimeError(f"provenance changed across cells: {field}")
        else:
            write_new(provenance_path, provenance_value)
        results = {}
        order = [args.variant] if args.variant else PROTOCOL["execution_order"]
        for variant in order:
            flags = PROTOCOL["variants"].get(variant, {})
            results[variant] = train_variant(
                api, args.dataset, config, data, args.seed, flags,
                initial,
                output / variant, variant
            )
        for variant in PROTOCOL["execution_order"]:
            path = output / variant / "result.json"
            if path.is_file() and variant not in results:
                results[variant] = json.loads(path.read_text())
        if set(results) != set(PROTOCOL["execution_order"]):
            print(json.dumps({
                "status": "independent_cells_completed_remaining_unstarted",
                "completed": sorted(results),
                "remaining": [name for name in PROTOCOL["execution_order"] if name not in results],
                "paper_goal_complete": False,
            }), flush=True)
            return 0
        full = results["full"]["metrics"]
        parent_result = results.get("parent_full")
        effects = {
            variant: {
                name: full[name] - result["metrics"][name]
                for name in ("auc", "acc", "nll", "brier", "ece_15bin")
            }
            for variant, result in results.items() if variant in PROTOCOL["variants"] and variant != "full"
        }
        module_pass = {
            name: effect["auc"] > 0 and effect["acc"] >= 0
            for name, effect in effects.items()
        }
        reference_pass = full["auc"] > PROTOCOL["reference_auc"][args.dataset]
        summary = {
            "status": "complete_independent_dataset_full",
            "dataset": args.dataset, "seed": args.seed, "results": results,
            "full_minus_ablation": effects,
            "full_minus_reference_auc": full["auc"] - PROTOCOL["reference_auc"][args.dataset],
            "module_point_pass": module_pass,
            "full_reference_point_pass": reference_pass,
            "full_minus_incumbent_auc": (
                full["auc"] - PROTOCOL["incumbent"]["auc"] if args.dataset == "assist2017" else None
            ),
            "full_minus_best_candidate_auc": (
                full["auc"] - PROTOCOL["best_candidate"]["auc"] if args.dataset == "assist2017" else None
            ),
            "best_candidate_role": PROTOCOL["best_candidate_role"],
            "full_minus_structural_control_auc": full["auc"] - PROTOCOL["structural_control"]["auc"],
            "full_minus_best_observed_auc": full["auc"] - PROTOCOL["best_observed_candidate"]["auc"],
            "dataset_point_admission_pass": reference_pass,
            "phase": PROTOCOL["phase"],
            "dataset_order": PROTOCOL["dataset_order"],
            "full_minus_parent_auc": (
                full["auc"] - parent_result["metrics"]["auc"]
                if parent_result is not None
                else None
            ),
            "all_point_gates_pass": False,
            "all_baseline_comparisons_complete": False,
            "paper_goal_complete": False,
            "reason_not_complete": "Only Assist2017 Full is executable here; an audited pass is required before separately frozen expansion and independent module controls. The same-five-of-eight goal remains incomplete.",
            "test_access": False, "window_test_access": False,
        }
        write_new(output / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    finally:
        stop.set()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
