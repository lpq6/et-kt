import copy
import unittest

import torch

from a2g.candidate import A2GModal
from a2g.modules.graph import DirectedConceptGraph
from test_model import aligned, build
from test_ffn_gate import activate_retained, activate_gates
from v27_audit_helpers import timed_batch


class LowMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_initial_state_and_rng_unchanged(self):
        original = build(A2GModal)
        original_rng = torch.get_rng_state().clone()
        lowmem = build(A2GModal, graph_checkpoint_steps=3)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(list(original.state_dict()), list(lowmem.state_dict()))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, lowmem.state_dict()[name]), name)

    def test_three_activated_training_steps_match(self):
        original = build(A2GModal)
        activate_retained(original)
        activate_gates(original)
        with torch.no_grad():
            original.ssm.input_memory.weight.fill_(0.02)
            original.ssm.mode_input_weight.normal_(std=0.01)
        lowmem = copy.deepcopy(original)
        lowmem.concept_graph.checkpoint_steps = 3
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=0.0001)
            for model in (original, lowmem)
        ]
        batch = timed_batch("cpu")
        for step in range(3):
            outputs, traces = [], []
            for model, optimizer in zip((original, lowmem), optimizers):
                optimizer.zero_grad(set_to_none=True)
                prediction, trace = aligned(model, batch)
                prediction.square().sum().backward()
                outputs.append(prediction)
                traces.append(trace)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            self.assertEqual(traces[0], traces[1])
            other = dict(lowmem.named_parameters())
            for name, parameter in original.named_parameters():
                if parameter.grad is None:
                    self.assertIsNone(other[name].grad)
                else:
                    torch.testing.assert_close(
                        parameter.grad,
                        other[name].grad,
                        rtol=0,
                        atol=0,
                        msg=f"{step}:{name}",
                    )
            for optimizer in optimizers:
                optimizer.step()
            for name, value in original.state_dict().items():
                torch.testing.assert_close(
                    value, lowmem.state_dict()[name], rtol=0, atol=0, msg=name
                )

    def test_bad_checkpoint_size_is_rejected(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                DirectedConceptGraph(32, 4, checkpoint_steps=value)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_graph_outputs_and_gradients_exact(self):
        from a2g.experiment import seed_all

        seed_all(42)
        original = build(A2GModal, device="cuda")
        activate_retained(original)
        lowmem = copy.deepcopy(original)
        lowmem.concept_graph.checkpoint_steps = 3
        batch = timed_batch("cuda")
        expected, _ = aligned(original, batch)
        actual, _ = aligned(lowmem, batch)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        expected.sum().backward()
        actual.sum().backward()
        parameters = dict(lowmem.named_parameters())
        for name, value in original.named_parameters():
            if value.grad is None:
                self.assertIsNone(parameters[name].grad)
            else:
                torch.testing.assert_close(
                    value.grad, parameters[name].grad, rtol=0, atol=0, msg=name
                )


if __name__ == "__main__":
    unittest.main()
