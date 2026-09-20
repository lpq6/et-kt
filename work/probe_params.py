import sys, torch, json, inspect
sys.path.insert(0, 'src')
from a2g.model import A2G
cfg = json.load(open('configs/assist2017_v43_lowmem.json'))
sig = inspect.signature(A2G.__init__)
valid = {k: v for k, v in cfg['model'].items() if k in sig.parameters}
valid.setdefault('n_question', 3109)
valid.setdefault('n_pid', 96)
m = A2G(**valid)
total = sum(p.numel() for p in m.parameters())
print('TOTAL params: {:,}  |  named_modules: {}'.format(total, len(list(m.named_modules()))))
top = {}
for name, mod in m.named_children():
    n = sum(p.numel() for p in mod.parameters())
    if n > 0:
        top[name] = n
for k, v in sorted(top.items(), key=lambda x: -x[1]):
    print('  {:24s} {:>10,d} ({:.1f}%)'.format(k, v, v / total * 100))
print()
t = cfg['training']
print('lr={}  wd={}  batch={}  max_epochs={}  patience={}'.format(
    t['learning_rate'], t['weight_decay'], t['batch_size'], t['max_epochs'], t['patience']))
