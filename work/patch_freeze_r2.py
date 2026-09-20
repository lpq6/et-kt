path = 'tools/freeze_versions.py'
src = open(path, encoding='utf-8').read()
entry = '''    "v46_compact_r2": {
        "architecture": "v46",
        "config": "assist2017_v46_compact.json",
        "checkpoint_steps": 16,
        "parent": "v46_compact",
        "description": "v46_compact refreshed with registered V46 training protocol (protocol.py) and config training_protocol id; shared weight-tied embeddings, AdamW wd=1e-4, seeded shuffle, 3-epoch warmup + cosine. Supersedes v46_compact (stale, pre-protocol). No performance claim.",
        "design_document": "V46_COMPACT.md",
    },'''
anchor = '    "v46_compact": {'
assert anchor in src, 'anchor not found'
src = src.replace(anchor, entry + '\n' + anchor, 1)
open(path, 'w', encoding='utf-8', newline='\n').write(src)
print('OK: v46_compact_r2 entry added')
