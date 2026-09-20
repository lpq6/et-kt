import io
from pathlib import Path
import sys
import unittest

import torch

from a2g.model import A2G
from a2g.randomness import A2GDropoutSubstreamAlignment


REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "provenance/server_172/independent_a2g_ffn_gate_20260911_v33"
)
sys.path.insert(0, str(REFERENCE / "sources"))
sys.path.insert(0, str(REFERENCE))
from ffn_gate_candidate import A2GMambaKT as ServerA2G
from test_ffn_gate import activate_retained, activate_gates
from v27_audit_helpers import timed_batch


FLAGS = [
    "use_evidence_branch",
    "use_ssm_branch",
    "use_attention_branch",
    "use_factorized_input",
    "use_item_attempt_stage",
    "use_history_pace",
    "use_prequential_newton",
    "use_concept_graph",
    "use_ffn_gate",
    "use_recency_weighted_concept_evidence",
    "use_split_boundary",
]


def build(cls=A2G, device="cpu", **flags):
    torch.manual_seed(42)
    model = cls(
        6,
        9,
        d_model=32,
        d_ff=64,
        n_blocks=4,
        num_attn_heads=4,
        dropout=0.2,
        seq_len=200,
        item_residual_dropout=0.4,
        **flags,
    ).to(device)
    with torch.no_grad():
        model.item_support_count.fill_(64)
        model.item_prior.weight.fill_(0.15)
        model.concept_prior.weight.fill_(-0.07)
    return model


def aligned(model, batch):
    align = A2GDropoutSubstreamAlignment(model, base_seed=42)
    try:
        align.begin_step(epoch=0, cursor=0)
        prediction = model(batch, train=True)[0]
        trace = align.end_step()
        return prediction, trace
    finally:
        if align._active:
            align.abort_step()
        align.close()


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_state_and_seed_stream_exact(self):
        server = build(ServerA2G)
        rng = torch.get_rng_state().clone()
        local = build()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(server.state_dict()), list(local.state_dict()))
        for name, value in server.state_dict().items():
            self.assertTrue(torch.equal(value, local.state_dict()[name]), name)

    def test_active_eval_and_each_ablation_equal_server(self):
        batch = timed_batch("cpu")
        for flag in [None] + FLAGS:
            with self.subTest(flag=flag):
                flags = {flag: 0} if flag else {}
                server, local = build(ServerA2G, **flags), build(A2G, **flags)
                activate_retained(server)
                activate_gates(server)
                local.load_state_dict(server.state_dict())
                server.eval()
                local.eval()
                torch.testing.assert_close(server(batch), local(batch), rtol=0, atol=0)

    def test_aligned_train_gradients_equal_server(self):
        batch = timed_batch("cpu")
        server, local = build(ServerA2G), build()
        activate_retained(server)
        activate_gates(server)
        local.load_state_dict(server.state_dict())
        expected, expected_trace = aligned(server, batch)
        actual, actual_trace = aligned(local, batch)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(expected_trace, actual_trace)
        expected.sum().backward()
        actual.sum().backward()
        other = dict(local.named_parameters())
        for name, value in server.named_parameters():
            if value.grad is None:
                self.assertIsNone(other[name].grad, name)
            else:
                torch.testing.assert_close(
                    value.grad, other[name].grad, rtol=0, atol=0, msg=name
                )

    def test_current_and_future_answers_do_not_leak(self):
        model = build().eval()
        activate_retained(model)
        activate_gates(model)
        batch = timed_batch("cpu")
        expected = model(batch)
        target = 3
        # Current/shifted inputs are overlapping views in the server fixture.
        altered = {name: value.clone() for name, value in batch.items()}
        altered["rseqs"][:, target:] = 1 - altered["rseqs"][:, target:]
        altered["shft_rseqs"][:] = 1 - altered["shft_rseqs"]
        altered["tseqs"][:, target:] += 9999999
        actual = model(altered)
        torch.testing.assert_close(
            expected[:, : target + 1], actual[:, : target + 1], rtol=0, atol=0
        )
        self.assertGreater(
            float((expected[:, target + 1 :] - actual[:, target + 1 :]).abs().max()), 0
        )

    def test_reload_and_batch_isolation(self):
        model = build().eval()
        activate_retained(model)
        batch = timed_batch("cpu")
        expected = model(batch)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = build().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(expected, restored(batch), rtol=0, atol=0)
        single = {name: value[:1] for name, value in batch.items()}
        torch.testing.assert_close(expected[:1], model(single), rtol=1e-5, atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward_finite(self):
        model = build(device="cuda")
        prediction = model(timed_batch("cuda"), train=True)[0]
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.sum().backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )


if __name__ == "__main__":
    unittest.main()
