import sys, json, torch
sys.path.insert(0, 'src')
from a2g.model import A2G

cfg = json.load(open('configs/assist2017_v46_compact.json'))
valid = {k: v for k, v in cfg['model'].items() if k in A2G.__init__.__code__.co_varnames}
valid.setdefault('n_question', 3109)
valid.setdefault('n_pid', 96)
m = A2G(**valid)
total = sum(p.numel() for p in m.parameters())
print('v46 shared-emb total params: {:,}'.format(total))
print('v43 baseline:                5,107,701')
print('reduction:                   {:.1f}%'.format((1 - total / 5107701) * 100))
print('named_modules:               {}'.format(len(list(m.named_modules()))))
print('bound concept_emb:', m.hist_concept_emb is m.concept_emb)
print('bound item_emb:  ', m.hist_item_emb is m.item_emb)
# 前向冒烟
batch = {
    'qseqs': torch.zeros(2, 8, dtype=torch.long),
    'cseqs': torch.zeros(2, 8, dtype=torch.long),
    'rseqs': torch.zeros(2, 8, dtype=torch.long),
    'shft_cseqs': torch.ones(2, 8, dtype=torch.long) * 3,
    'shft_qseqs': torch.ones(2, 8, dtype=torch.long) * 5,
    'shft_rseqs': torch.ones(2, 8, dtype=torch.long),
    'smasks': torch.ones(2, 8, dtype=torch.bool),
}
m.eval()
with torch.no_grad():
    out = m(batch)
print('forward shape:', tuple(out.shape), '| finite:', bool(torch.isfinite(out).all()))
print('OK')
