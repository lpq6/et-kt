path = 'src/a2g/experiment.py'
src = open(path, encoding='utf-8').read()

# 1) Adam -> AdamW(config) + warmup/cosine scheduler
old = """    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )"""
new = """    opt_name = config["training"].get("optimizer", "adam")
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

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)"""
assert old in src, 'optimizer block not found'
src = src.replace(old, new)

# 2) train loader: pass shuffle from config (default False -> old behavior)
old2 = """                    loader(bundles["train"], batch_size, epoch - 1)"""
new2 = """                    loader(
                        bundles["train"],
                        batch_size,
                        epoch - 1,
                        shuffle=config["training"].get("shuffle", False),
                    )"""
assert old2 in src, 'train loader call not found'
src = src.replace(old2, new2)

# 3) scheduler.step() after each epoch (before patience check)
old3 = """                if epoch - best_epoch >= config["training"]["patience"]:
                    break"""
new3 = """                if scheduler is not None:
                    scheduler.step()
                if epoch - best_epoch >= config["training"]["patience"]:
                    break"""
assert old3 in src, 'patience block not found'
src = src.replace(old3, new3)

# 4) loader() signature: accept shuffle and forward to sampler
old4 = """def loader(bundle, size, epoch=0):
    sampler = FixedOrderBatchSampler(bundle["metadata"], size, seed=42, epoch=epoch)"""
new4 = """def loader(bundle, size, epoch=0, shuffle=False):
    sampler = FixedOrderBatchSampler(
        bundle["metadata"], size, seed=42, epoch=epoch, shuffle=shuffle
    )"""
assert old4 in src, 'loader signature not found'
src = src.replace(old4, new4)

# 5) ensure math imported
if 'import math\n' not in src:
    anchor = 'import json\n'
    assert anchor in src
    src = src.replace(anchor, anchor + 'import math\n', 1)

open(path, 'w', encoding='utf-8', newline='\n').write(src)
print('OK: protocol upgrade applied')
