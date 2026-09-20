path = 'tools/freeze_versions.py'
src = open(path, encoding='utf-8').read()
entry = '''    "v46_compact": {
        "architecture": "v46",
        "config": "assist2017_v46_compact.json",
        "checkpoint_steps": 16,
        "parent": "v43_lowmem_r3",
        "description": "v43 architecture with shared (weight-tied) target/history embeddings (16 pct fewer params) plus upgraded protocol: AdamW wd=1e-4, seeded per-epoch shuffle, 3-epoch warmup + cosine decay. Keeps all v43 modules; efficiency/generalization target, no performance claim.",
        "design_document": "V46_COMPACT.md",
    },'''
anchor = '    "v45_umk_r2": {'
assert anchor in src, 'anchor not found'
src = src.replace(anchor, entry + '\n' + anchor, 1)
open(path, 'w', encoding='utf-8', newline='\n').write(src)
print('OK: v46_compact entry added')
