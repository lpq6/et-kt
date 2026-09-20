from pathlib import Path
import sys
import unittest

import torch

from a2g.candidate import A2GModal
from a2g.model import A2G
from a2g.modules.input_memory import CausalInputMemory
from test_model import build, aligned
from v27_audit_helpers import timed_batch


V43 = (
    Path(__file__).resolve().parents[1]
    / "provenance/server_172/independent_a2g_modal_residual_20260913_v43"
)
sys.path.append(str(V43))
from modal_residual_candidate import A2GMambaKT as ServerModal


class CandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_v43_state_and_initial_predictions_match_server(self):
        server, local = build(ServerModal), build(A2GModal)
        self.assertEqual(list(server.state_dict()), list(local.state_dict()))
        for key, value in server.state_dict().items():
            self.assertTrue(torch.equal(value, local.state_dict()[key]), key)
        expected, trace = aligned(server, timed_batch("cpu"))
        actual, local_trace = aligned(local, timed_batch("cpu"))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(trace, local_trace)
        expected.sum().backward()
        actual.sum().backward()
        local_params = dict(local.named_parameters())
        for key, parameter in server.named_parameters():
            if parameter.grad is None:
                self.assertIsNone(local_params[key].grad)
            else:
                torch.testing.assert_close(
                    parameter.grad, local_params[key].grad, rtol=0, atol=0, msg=key
                )

    def test_disabled_candidate_equals_reference(self):
        reference = build(A2G).eval()
        candidate = build(A2GModal, use_input_memory=0, use_multimode_residual=0).eval()
        torch.testing.assert_close(
            reference(timed_batch("cpu")), candidate(timed_batch("cpu")), rtol=0, atol=0
        )

    def test_activated_candidate_matches_server_and_is_causal(self):
        server, local = build(ServerModal).eval(), build(A2GModal).eval()
        with torch.no_grad():
            server.ssm.input_memory.weight.fill_(0.04)
            server.ssm.mode_input_weight.normal_(std=0.01)
            server.ssm.mode_read_weight.normal_(std=0.01)
        local.load_state_dict(server.state_dict())
        batch = timed_batch("cpu")
        expected = local(batch)
        torch.testing.assert_close(server(batch), expected, rtol=0, atol=0)
        batch = {key: value.clone() for key, value in batch.items()}
        batch["rseqs"][:, 3:] = 1 - batch["rseqs"][:, 3:]
        torch.testing.assert_close(expected[:, :4], local(batch)[:, :4], rtol=0, atol=0)

    def test_input_memory_never_reads_current_or_future_projection(self):
        memory = CausalInputMemory(3)
        with torch.no_grad():
            memory.weight.fill_(1)
        inputs = torch.arange(18, dtype=torch.float32).view(1, 6, 3)
        output = memory(inputs)
        for target in range(6):
            torch.testing.assert_close(
                output[:, target], inputs[:, max(0, target - 4) : target].sum(1)
            )


if __name__ == "__main__":
    unittest.main()
