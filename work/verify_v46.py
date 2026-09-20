import sys, json, torch
sys.path.insert(0, 'src')
from a2g.model import A2G
from a2g.data import FixedOrderBatchSampler

# 1) 共享嵌入模型构建
cfg = json.load(open('configs/assist2017_v46_compact.json'))
valid = {k: v for k, v in cfg['model'].items() if k in A2G.__init__.__code__.co_varnames}
valid.setdefault('n_question', 3109)
valid.setdefault('n_pid', 96)
m = A2G(**valid)
total = sum(p.numel() for p in m.parameters())
print('v46 total params: {:,}'.format(total))
print('v43 baseline:     5,107,701')
print('reduction:        {:.1f}%'.format((1 - total / 5107701) * 100))
print('named_modules:    {}'.format(len(list(m.named_modules()))))
# 绑定检查
print('hist==cur concept_emb:', m.hist_concept_emb is m.concept_emb)
print('hist==cur item_emb:  ', m.hist_item_emb is m.item_emb)
sd = m.state_dict()
print('state_dict keys:', len(sd), '| has hist_concept_emb.weight:', 'hist_concept_emb.weight' in sd)

# 2) sampler shuffle 行为
class E:
    def __init__(self, i): self.csv_row_index = i; self.learner_uid = i % 7
meta = [E(i) for i in range(100)]
s1 = FixedOrderBatchSampler(meta, 32, seed=42, epoch=1, shuffle=True)
s2 = FixedOrderBatchSampler(meta, 32, seed=42, epoch=1, shuffle=True)
s3 = FixedOrderBatchSampler(meta, 32, seed=42, epoch=2, shuffle=True)
b1, b2, b3 = list(s1), list(s2), list(s3)
print('same seed/epoch deterministic:', b1 == b2)
print('different epoch differs:      ', b1 != b3)
print('batch sizes:', [len(b) for b in b1])
s_old = FixedOrderBatchSampler(meta, 32, seed=42, epoch=1)
print('shuffle=False keeps order:    ', list(s_old)[0] == list(range(32)))
print('ALL CHECKS PASSED')
