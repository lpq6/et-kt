import sys, torch, json, inspect
sys.path.insert(0, 'src')
from a2g.model import A2G
cfg = json.load(open('configs/assist2017_v46_compact.json'))
sig = inspect.signature(A2G.__init__)
valid = {k: v for k, v in cfg['model'].items() if k in sig.parameters}
valid.setdefault('n_question', 3109)
valid.setdefault('n_pid', 96)
print('model keys passed:', sorted(valid.keys()))
m = A2G(**valid)
total = sum(p.numel() for p in m.parameters())
print('TOTAL params: {:,}  |  named_modules: {}'.format(total, len(list(m.named_modules()))))
print('--- top-level modules (named_children) ---')
top = {}
for name, mod in m.named_children():
    n = sum(p.numel() for p in mod.parameters())
    if n > 0:
        top[name] = n
for k, v in sorted(top.items(), key=lambda x: -x[1]):
    print('  {:28s} {:>10,d} ({:.1f}%)'.format(k, v, v / total * 100))
print('--- named_modules detail (all) ---')
for i, (name, mod) in enumerate(m.named_modules()):
    n = sum(p.numel() for p in mod.parameters())
    print('  [{:2d}] {:55s} {:>9,d}'.format(i, name, n))
