import sys, json, torch, inspect
sys.path.insert(0, 'src')
from a2g.experiment import make_model, ABLATIONS
cfg = json.load(open('configs/assist2017_v46_ablation.json'))
cfg['model'].update(d_model=64, d_ff=128, num_attn_heads=4)
cfg['data'].update(num_c=16, num_q=32)
print('ABLATIONS total:', len(ABLATIONS))
NEW = ['no_split_boundary', 'no_rwce', 'no_memory_readout', 'no_shared_embeddings', 'no_evidence_equivalent_residual']
for v in ['full'] + NEW:
    torch.manual_seed(42)
    m = make_model(json.loads(json.dumps(cfg)), v)
    n = sum(p.numel() for p in m.parameters())
    keys = list(m.state_dict())
    print(f'{v:32s} params={n:>10,d}  state_keys={len(keys):>3d}  attr={ABLATIONS.get(v,"-")}')
    # forward smoke
    B, L = 2, 8
    xq = torch.randint(1, 33, (B, L))
    xc = torch.randint(0, 16, (B, L))
    r = torch.randint(0, 3, (B, L))
    tq = torch.randint(1, 33, (B,))
    tc = torch.randint(0, 16, (B,))
    try:
        out = m(xq, xc, r, tq, tc)
        print(f'   forward ok: out={tuple(out.shape)} finite={torch.isfinite(out).all().item()}')
    except Exception as e:
        print(f'   forward FAIL: {type(e).__name__}: {str(e)[:120]}')
