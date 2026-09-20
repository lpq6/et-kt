path = 'tools/freeze_versions.py'
src = open(path, encoding='utf-8').read()
entry = '''    "v46_compact_r3": {
        "architecture": "v46",
        "config": "assist2017_v46_compact.json",
        "checkpoint_steps": 16,
        "parent": "v46_compact_r2",
        "description": "v46_compact_r2 plus make_model architecture routing for v46 (A2GModal). Complete runnable v46 candidate: shared weight-tied embeddings (-16 pct params), AdamW wd=1e-4, seeded shuffle, 3-epoch warmup + cosine, V46 protocol id. Supersedes stale v46_compact and non-runnable v46_compact_r2. No performance claim.",
        "design_document": "V46_COMPACT.md",
    },'''
anchor = '    "v46_compact_r2": {'
assert anchor in src, 'anchor not found'
src = src.replace(anchor, entry + '\n' + anchor, 1)
open(path, 'w', encoding='utf-8', newline='\n').write(src)
print('OK: v46_compact_r3 entry added')
