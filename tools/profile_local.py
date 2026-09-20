"""Bounded production-shape training step, not a performance experiment."""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

from a2g.experiment import load_config, make_model, seed_all
from a2g.provenance import sha256, source_hashes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/assist2017_v43.json")
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 < args.memory_fraction <= 1:
        parser.error("batch size must be positive and memory fraction in (0,1]")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    seed_all(42)
    config = load_config(args.config)
    batch_size, length = args.batch_size, config["data"]["maxlen"]
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    free_before, total_before = torch.cuda.mem_get_info()
    model = make_model(config, "full").cuda()
    with torch.no_grad():
        model.item_support_count.fill_(64)
    questions = torch.randint(
        2, config["data"]["num_q"], (batch_size, length), device="cuda"
    )
    concepts = torch.randint(
        2, config["data"]["num_c"], (batch_size, length), device="cuda"
    )
    responses = torch.randint(0, 2, (batch_size, length), device="cuda")
    times = (
        torch.arange(length, device="cuda", dtype=torch.long)[None].expand(
            batch_size, -1
        )
        * 1000
    )
    batch = {}
    for field, tensor in [
        ("qseqs", questions),
        ("cseqs", concepts),
        ("rseqs", responses),
        ("tseqs", times),
    ]:
        batch[field], batch["shft_" + field] = tensor[:, :-1], tensor[:, 1:]
    batch["smasks"] = torch.ones(
        batch_size, length - 1, device="cuda", dtype=torch.bool
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    stage = "forward"
    print(
        f"Profiling direct batch={batch_size}, length={length}, "
        f"memory_fraction={args.memory_fraction}",
        flush=True,
    )
    try:
        prediction = model(batch, train=True)[0]
        loss = F.binary_cross_entropy(
            prediction[:, 1:].double(), batch["shft_rseqs"].double()
        )
        stage = "backward"
        loss.backward()
        stage = "optimizer_step"
        optimizer.step()
        torch.cuda.synchronize()
        result = {
            "status": "synthetic_step_complete",
            "loss": float(loss),
            "parameters": sum(p.numel() for p in model.parameters()),
            "finite_gradients": all(
                bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
                if p.grad is not None
            ),
        }
    except torch.cuda.OutOfMemoryError as error:
        result = {
            "status": "cuda_out_of_memory",
            "failed_stage": stage,
            "error": str(error),
        }
    result.update(
        batch_size=batch_size,
        sequence_length=length,
        architecture=config["architecture"],
        elapsed_seconds=time.monotonic() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu=torch.cuda.get_device_name(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        memory_fraction=args.memory_fraction,
        allocator_budget_bytes=int(args.memory_fraction * total_before),
        free_device_bytes_before_model=free_before,
        precision="float32_model_float64_bce",
        gradient_accumulation_steps=1,
        microbatching=False,
        config_sha256=sha256(args.config),
        model_configuration=config["model"],
        source=source_hashes(Path(__file__).resolve().parents[1] / "src/a2g"),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        auc_claim=False,
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
