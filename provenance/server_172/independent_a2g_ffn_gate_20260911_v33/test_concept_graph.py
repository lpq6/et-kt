"""Independent graph/GRU equations, structural witnesses and V28 parity."""

import ast
import copy
import math
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT))
from concept_graph_candidate import A2GMambaKT as Candidate, DirectedConceptGraph, graph_state_rollout
from prequential_newton_candidate import A2GMambaKT as V28
from v19_audit_helpers import aligned, build, common_gradients_match
from v27_audit_helpers import timed_batch


GRAPH_SUFFIXES = sorted((
    "source_projection.weight", "destination_projection.weight",
    "event_projection.weight", "event_projection.bias", "state_projection.weight",
    "cell.weight_ih", "cell.weight_hh", "cell.bias_ih", "cell.bias_hh", "output.weight",
))


def synthetic_graph(batch=2, length=7, width=8, nodes=4, dtype=torch.float64, device="cpu"):
    generator = torch.Generator().manual_seed(20260911)
    history = torch.randn(batch, length, width, generator=generator, dtype=dtype)
    table = torch.randn(nodes + 2, width, generator=generator, dtype=dtype)
    index = torch.arange(batch * length).reshape(batch, length)
    source = 2 + index % nodes
    source[:, 0] = 0
    response = (index // 2) % 2
    response[:, 0] = 2
    target = 2 + (index + 1) % nodes
    return tuple(value.to(device) for value in (history, source, response, target, table))


@torch.no_grad()
def activate_graph(module):
    value = module.output.weight
    index = torch.arange(value.numel(), device=value.device, dtype=value.dtype)
    value.copy_((0.07 * torch.sin(index + 1)).reshape_as(value))


def linear_oracle(value, weight, bias=None):
    rows = [
        (value * row).sum(-1) + (0 if bias is None else bias[index])
        for index, row in enumerate(weight)
    ]
    return torch.stack(rows, dim=-1)


def layer_norm_oracle(value):
    centered = value - value.mean(-1, keepdim=True)
    return centered / (centered.square().mean(-1, keepdim=True) + 1e-5).sqrt()


def edge_oracle(module, table):
    nodes = table.size(0) - 2
    if nodes <= 1:
        return table.new_ones(nodes, nodes)
    z = layer_norm_oracle(table[2:])
    source = linear_oracle(z, module.source_projection.weight)
    destination = linear_oracle(z, module.destination_projection.weight)
    rows = []
    for sender in range(nodes):
        off_diagonal = [receiver for receiver in range(nodes) if sender != receiver]
        scores = torch.stack([
            (source[sender] * destination[receiver]).sum() / math.sqrt(module.rank)
            for receiver in off_diagonal
        ])
        probabilities = torch.exp(scores - torch.logsumexp(scores, dim=0))
        cursor = 0
        row = []
        for receiver in range(nodes):
            if receiver == sender:
                row.append(table.new_ones(()))
            else:
                row.append(probabilities[cursor])
                cursor += 1
        rows.append(torch.stack(row))
    return torch.stack(rows)


def gru_oracle(message, state, cell):
    input_gates = linear_oracle(message, cell.weight_ih, cell.bias_ih)
    hidden_gates = linear_oracle(state, cell.weight_hh, cell.bias_hh)
    ir, iz, inn = input_gates.chunk(3, dim=-1)
    hr, hz, hn = hidden_gates.chunk(3, dim=-1)
    reset = torch.sigmoid(ir + hr)
    update = torch.sigmoid(iz + hz)
    proposal = torch.tanh(inn + reset * hn)
    return (1 - update) * proposal + update * state


def rollout_oracle(events, hist_c, hist_r, target_c, edges, cell, state_projection):
    batch, length, rank = events.shape
    nodes = edges.size(0)
    if nodes == 0:
        return torch.zeros_like(events)
    learners = []
    for learner in range(batch):
        state = [events.new_zeros(rank) for _ in range(nodes)]
        reads = []
        for position in range(length):
            sender = int(hist_c[learner, position]) - 2
            response = int(hist_r[learner, position])
            if 0 <= sender < nodes and response in (0, 1):
                message = torch.tanh(
                    events[learner, position]
                    + linear_oracle(state[sender], state_projection.weight)
                )
                state = [
                    (1 - edges[sender, receiver]) * previous
                    + edges[sender, receiver] * gru_oracle(message, previous, cell)
                    for receiver, previous in enumerate(state)
                ]
            receiver = int(target_c[learner, position]) - 2
            reads.append(state[receiver] if 0 <= receiver < nodes else events.new_zeros(rank))
        learners.append(torch.stack(reads))
    return torch.stack(learners)


def direct_graph(module, history, hist_c, hist_r, target_c, table):
    events = linear_oracle(
        layer_norm_oracle(history), module.event_projection.weight, module.event_projection.bias,
    )
    reads = rollout_oracle(
        events, hist_c, hist_r, target_c, edge_oracle(module, table),
        module.cell, module.state_projection,
    )
    return linear_oracle(reads, module.output.weight)


def forward_scope_matches_v28():
    def function(filename):
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "A2GMambaKT")
        return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")

    base, candidate = function("prequential_newton_candidate.py"), function("concept_graph_candidate.py")
    for node, flag in ((base, "use_prequential_newton"), (candidate, "use_concept_graph")):
        guard = node.body.pop(0)
        expected = ast.parse(f"if not self.{flag}:\n    return super().forward(dcur, train=train, qtest=qtest)").body[0]
        if ast.dump(guard) != ast.dump(expected):
            return False
    inserted, conditional = 0, 0
    for node in candidate.body:
        if not isinstance(node, ast.Try):
            continue
        keep = []
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign) and isinstance(statement.value, ast.BinOp)
                and isinstance(statement.value.right, ast.Call)
                and isinstance(statement.value.right.func, ast.Attribute)
                and statement.value.right.func.attr == "concept_graph"
            ):
                expected = ast.parse(
                    "token = token + self.concept_graph(history, hist_c, hist_r, target_c, "
                    "self.hist_concept_emb.weight[:self.n_question])"
                ).body[0]
                if ast.dump(statement) != ast.dump(expected):
                    return False
                inserted += 1
                continue
            if (
                isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "correction"
            ):
                value = statement.value
                if not isinstance(value, ast.IfExp):
                    return False
                if (
                    ast.dump(value.test) != ast.dump(ast.parse("self.use_prequential_newton", mode="eval").body)
                    or ast.dump(value.orelse) != ast.dump(ast.parse("torch.zeros_like(base_logits)", mode="eval").body)
                ):
                    return False
                statement.value = value.body
                conditional += 1
            keep.append(statement)
        node.body = keep
    return inserted == conditional == 1 and ast.dump(base) == ast.dump(candidate)


class GraphEquationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.default_generator.manual_seed(42)
        self.module = DirectedConceptGraph(8, 2).double()
        self.values = synthetic_graph()

    def test_ten_named_parameters_and_only_output_is_zero(self):
        self.assertEqual(sorted(dict(self.module.named_parameters())), GRAPH_SUFFIXES)
        for name, value in self.module.named_parameters():
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertEqual(bool(value.eq(0).all()), name == "output.weight")

    def test_small_and_production_parameter_counts(self):
        self.assertEqual(sum(p.numel() for p in DirectedConceptGraph(32, 4).parameters()), 1528)
        self.assertEqual(sum(p.numel() for p in DirectedConceptGraph(256, 8).parameters()), 40160)

    def test_zero_output_at_initialization(self):
        self.assertTrue(bool(self.module(*self.values).eq(0).all()))

    def test_edges_match_independent_sender_receiver_formula(self):
        actual = self.module.edge_weights(self.values[-1])
        expected = edge_oracle(self.module, self.values[-1])
        self.assertTrue(torch.allclose(actual, expected, atol=2e-14, rtol=2e-14))
        self.assertFalse(torch.allclose(actual, actual.T, atol=1e-6, rtol=1e-6))

    def test_self_weights_one_neighbors_sum_one(self):
        edges = self.module.edge_weights(self.values[-1])
        self.assertTrue(torch.equal(edges.diag(), torch.ones(4, dtype=edges.dtype)))
        self.assertTrue(torch.allclose(edges.sum(-1), torch.full((4,), 2.0, dtype=edges.dtype)))
        self.assertTrue(bool(((edges >= 0) & (edges <= 1)).all()))

    def test_padding_and_unknown_embeddings_do_not_define_edges(self):
        before = self.module.edge_weights(self.values[-1])
        changed = self.values[-1].clone()
        changed[:2] *= 1e6
        self.assertTrue(torch.equal(before, self.module.edge_weights(changed)))

    def test_single_known_node_has_only_self_edge(self):
        values = synthetic_graph(nodes=1)
        self.assertTrue(torch.equal(self.module.edge_weights(values[-1]), torch.ones(1, 1, dtype=torch.float64)))
        activate_graph(self.module)
        self.assertTrue(torch.allclose(self.module(*values), direct_graph(self.module, *values), atol=2e-12, rtol=2e-12))

    def test_empty_known_vocabulary_has_zero_features(self):
        values = list(self.values)
        values[-1] = values[-1][:2]
        activate_graph(self.module)
        self.assertTrue(bool(self.module(*values).eq(0).all()))

    def test_full_output_matches_independent_gru_and_graph_equations(self):
        activate_graph(self.module)
        self.assertTrue(torch.allclose(self.module(*self.values), direct_graph(self.module, *self.values), atol=3e-12, rtol=3e-12))

    def test_all_input_and_parameter_gradients_match_oracle(self):
        activate_graph(self.module)
        values = list(self.values)
        values[0] = values[0].clone().requires_grad_()
        values[-1] = values[-1].clone().requires_grad_()
        parameters = (values[0], values[-1], *self.module.parameters())
        actual = self.module(*values).square().sum()
        expected = direct_graph(self.module, *values).square().sum()
        left = torch.autograd.grad(actual, parameters, retain_graph=True)
        right = torch.autograd.grad(expected, parameters)
        for a, b in zip(left, right):
            self.assertTrue(torch.allclose(a, b, atol=3e-10, rtol=3e-10))

    def test_output_projection_receives_gradient_at_zero(self):
        self.module(*self.values).sum().backward()
        self.assertGreater(float(self.module.output.weight.grad.abs().sum()), 0)

    def test_inner_parameters_have_zero_gradient_at_zero_output(self):
        self.module(*self.values).sum().backward()
        for name, value in self.module.named_parameters():
            if name != "output.weight":
                self.assertIsNotNone(value.grad)
                self.assertTrue(bool(value.grad.eq(0).all()), name)

    def test_every_parameter_has_finite_nonzero_gradient_after_activation(self):
        activate_graph(self.module)
        self.module(*self.values).square().sum().backward()
        for name, value in self.module.named_parameters():
            self.assertTrue(bool(torch.isfinite(value.grad).all()), name)
            self.assertGreater(float(value.grad.abs().sum()), 0, name)

    def test_first_prediction_without_history_is_zero(self):
        activate_graph(self.module)
        self.assertTrue(bool(self.module(*self.values)[:, 0].eq(0).all()))

    def test_every_prefix_matches_independent_recurrence(self):
        activate_graph(self.module)
        before = self.module(*self.values)
        for length in range(1, self.values[0].size(1) + 1):
            prefix = [value[:, :length] for value in self.values[:4]] + [self.values[-1]]
            actual = self.module(*prefix)
            self.assertTrue(torch.allclose(actual, before[:, :length], atol=3e-12, rtol=3e-12))
            self.assertTrue(torch.allclose(actual, direct_graph(self.module, *prefix), atol=3e-12, rtol=3e-12))

    def test_future_inputs_do_not_change_prefix(self):
        activate_graph(self.module)
        before = self.module(*self.values)
        for start in range(1, self.values[0].size(1)):
            changed = [value.clone() for value in self.values]
            changed[0][:, start:] *= -7
            changed[1][:, start:] = 3
            changed[2][:, start:] = 1 - changed[2][:, start:]
            changed[3][:, start:] = 4
            self.assertTrue(torch.equal(before[:, :start], self.module(*changed)[:, :start]))

    def test_invalid_history_is_an_identity_update(self):
        activate_graph(self.module)
        values = [value.clone() for value in self.values]
        values[1][:, 2] = 1
        values[2][:, 4] = 2
        before = self.module(*values)
        values[0][:, [2, 4]] *= -200
        self.assertTrue(torch.equal(before, self.module(*values)))

    def test_all_invalid_history_stays_zero_despite_gru_biases(self):
        values = list(self.values)
        values[2] = torch.full_like(values[2], 2)
        activate_graph(self.module)
        self.assertTrue(bool(self.module(*values).eq(0).all()))

    def test_padding_unknown_and_out_of_vocabulary_target_reads_are_zero(self):
        activate_graph(self.module)
        values = [value.clone() for value in self.values]
        values[3][:, 2:5] = torch.tensor([0, 1, 999])
        self.assertTrue(bool(self.module(*values)[:, 2:5].eq(0).all()))

    def test_learner_isolation(self):
        activate_graph(self.module)
        before = self.module(*self.values)[0]
        changed = [value.clone() for value in self.values]
        changed[0][1] *= -4
        self.assertTrue(torch.equal(before, self.module(*changed)[0]))

    def test_learner_permutation(self):
        activate_graph(self.module)
        changed = [value.flip(0) for value in self.values[:4]] + [self.values[-1]]
        self.assertTrue(torch.allclose(self.module(*self.values).flip(0), self.module(*changed), atol=3e-12, rtol=3e-12))

    def test_node_relabeling_equivariance(self):
        activate_graph(self.module)
        values = [value.clone() for value in self.values]
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        values[-1][2:] = self.values[-1][2:][permutation]
        for index in (1, 3):
            known = values[index].ge(2)
            values[index][known] = inverse[values[index][known] - 2] + 2
        self.assertTrue(torch.allclose(self.module(*self.values), self.module(*values), atol=3e-12, rtol=3e-12))

    def test_no_state_carry_buffers_dropout_or_rng_consumption(self):
        activate_graph(self.module)
        before = self.module(*self.values)
        state = copy.deepcopy(self.module.state_dict())
        rng = torch.get_rng_state().clone()
        self.module(*synthetic_graph(length=5))
        self.assertTrue(torch.equal(before, self.module(*self.values)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, self.module.state_dict()[name]) for name, value in state.items()))
        self.assertEqual(list(self.module.buffers()), [])
        self.assertFalse(any(isinstance(module, torch.nn.Dropout) for module in self.module.modules()))

    def test_training_mode_does_not_change_graph_module(self):
        activate_graph(self.module)
        before = self.module(*self.values)
        self.module.eval()
        self.assertTrue(torch.equal(before, self.module(*self.values)))

    def test_large_finite_inputs_have_finite_output_and_gradients(self):
        activate_graph(self.module)
        values = [value.clone() for value in self.values]
        values[0] = (values[0] * 1e6).requires_grad_()
        values[-1] *= 1e6
        result = self.module(*values)
        result.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(bool(torch.isfinite(values[0].grad).all()))

    def test_invalid_width_and_head_count_rejected(self):
        for width, heads in ((0, 2), (8, 0), (7, 2)):
            with self.assertRaises(ValueError):
                DirectedConceptGraph(width, heads)

    def test_misaligned_response_shape_rejected(self):
        values = list(self.values)
        values[2] = values[2][:, :-1]
        with self.assertRaises(ValueError):
            self.module(*values)

    def test_float_concept_ids_rejected(self):
        values = list(self.values)
        values[1] = values[1].float()
        with self.assertRaises(ValueError):
            self.module(*values)

    def test_wrong_concept_embedding_width_rejected(self):
        values = list(self.values)
        values[-1] = values[-1][:, :-1]
        with self.assertRaises(ValueError):
            self.module(*values)


class StructuralWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.default_generator.manual_seed(42)
        self.module = DirectedConceptGraph(8, 2).double()
        self.edges = torch.tensor([[1., .5, 0.], [0., 1., .5], [0., 0., 1.]], dtype=torch.float64)

    def run_states(self, events, source, target, edges=None):
        return graph_state_rollout(
            events, source, torch.ones_like(source), target,
            self.edges if edges is None else edges, self.module.cell, self.module.state_projection,
        )

    def test_unpracticed_target_changes_only_through_incoming_edges(self):
        events = torch.ones(1, 2, 4, dtype=torch.float64)
        source = torch.tensor([[2, 2]])
        target = torch.tensor([[3, 3]])
        connected = self.run_states(events, source, target)
        isolated = self.run_states(events, source, target, torch.eye(3, dtype=torch.float64))
        self.assertGreater(float(connected.abs().sum()), 0)
        self.assertTrue(bool(isolated.eq(0).all()))

    def test_source_state_enables_ordered_two_hop_propagation(self):
        events = torch.tensor([[[.1, -.2, .3, -.4], [.4, .3, .2, .1]]], dtype=torch.float64)
        changed = events.clone()
        changed[:, 0] *= -5
        source, target = torch.tensor([[2, 3]]), torch.tensor([[4, 4]])
        first, second = self.run_states(events, source, target), self.run_states(changed, source, target)
        self.assertTrue(bool(first[:, 0].eq(0).all()))
        self.assertTrue(bool(second[:, 0].eq(0).all()))
        self.assertFalse(torch.allclose(first[:, 1], second[:, 1], atol=1e-10, rtol=1e-10))
        with torch.no_grad():
            self.module.state_projection.weight.zero_()
        self.assertTrue(torch.equal(self.run_states(events, source, target), self.run_states(changed, source, target)))

    def test_same_outcome_counts_and_last_event_can_have_different_states(self):
        source = torch.tensor([[2, 3, 2, 3, 2]])
        response = torch.tensor([[1, 0, 0, 1, 1]])
        permutation = torch.tensor([1, 0, 3, 2, 4])
        generator = torch.Generator().manual_seed(17)
        events = torch.randn(1, 5, 4, generator=generator, dtype=torch.float64)
        target = torch.full_like(source, 4)
        edges = torch.tensor([[1., .5, .5], [.5, 1., .5], [.5, .5, 1.]], dtype=torch.float64)
        before = graph_state_rollout(events, source, response, target, edges, self.module.cell, self.module.state_projection)
        after = graph_state_rollout(
            events[:, permutation], source[:, permutation], response[:, permutation], target, edges,
            self.module.cell, self.module.state_projection,
        )
        self.assertTrue(torch.equal(
            torch.bincount((source * 2 + response).flatten()),
            torch.bincount((source[:, permutation] * 2 + response[:, permutation]).flatten()),
        ))
        self.assertTrue(torch.equal(events[:, -1], events[:, permutation][:, -1]))
        self.assertFalse(torch.allclose(before[:, -1], after[:, -1], atol=1e-10, rtol=1e-10))

    def test_identity_edges_do_not_update_other_nodes(self):
        events = torch.ones(1, 3, 4, dtype=torch.float64)
        source = torch.tensor([[2, 2, 2]])
        target = torch.tensor([[3, 4, 3]])
        self.assertTrue(bool(self.run_states(events, source, target, torch.eye(3, dtype=torch.float64)).eq(0).all()))


class ModelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_graph_insertion_and_newton_conditional_differ_in_forward_ast(self):
        self.assertTrue(forward_scope_matches_v28())

    def test_common_initial_state_and_rng_match_v28(self):
        parent = build(V28, "cpu")
        rng = torch.get_rng_state().clone()
        candidate = build(Candidate, "cpu")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(parent.state_dict()), 95)
        self.assertEqual(len(candidate.state_dict()), 105)
        self.assertEqual(
            sorted(set(candidate.state_dict()) - set(parent.state_dict())),
            ["concept_graph." + name for name in GRAPH_SUFFIXES],
        )
        self.assertTrue(all(torch.equal(value, candidate.state_dict()[name]) for name, value in parent.state_dict().items()))

    def test_initial_eval_outputs_match_v28(self):
        parent, candidate = build(V28, "cpu").eval(), build(Candidate, "cpu").eval()
        batch = timed_batch(torch.device("cpu"))
        with torch.no_grad():
            self.assertTrue(torch.equal(parent(batch), candidate(batch)))

    def test_initial_train_outputs_common_gradients_and_dropout_trace_match_v28(self):
        parent, candidate = build(V28, "cpu").train(), build(Candidate, "cpu").train()
        batch = timed_batch(torch.device("cpu"))
        old, old_trace = aligned(parent, batch)
        new, new_trace = aligned(candidate, batch)
        self.assertTrue(torch.equal(old, new))
        self.assertEqual(old_trace, new_trace)
        for prediction in (old, new):
            F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
        self.assertTrue(common_gradients_match(parent, candidate))
        self.assertGreater(float(candidate.concept_graph.output.weight.grad.abs().sum()), 0)

    def test_disabled_nonzero_graph_matches_v28_and_has_no_gradient(self):
        parent = build(V28, "cpu").train()
        candidate = build(Candidate, "cpu", use_concept_graph=0).train()
        activate_graph(candidate.concept_graph)
        batch = timed_batch(torch.device("cpu"))
        old, old_trace = aligned(parent, batch)
        new, new_trace = aligned(candidate, batch)
        self.assertTrue(torch.equal(old, new))
        self.assertEqual(old_trace, new_trace)
        for prediction in (old, new):
            F.binary_cross_entropy(prediction[:, 1:], batch["shft_rseqs"].float()).backward()
        self.assertTrue(common_gradients_match(parent, candidate))
        self.assertTrue(all(value.grad is None for value in candidate.concept_graph.parameters()))

    def test_active_graph_changes_predictions(self):
        parent, candidate = build(V28, "cpu").eval(), build(Candidate, "cpu").eval()
        activate_graph(candidate.concept_graph)
        batch = timed_batch(torch.device("cpu"))
        with torch.no_grad():
            self.assertFalse(torch.equal(parent(batch), candidate(batch)))

    def test_graph_node_count_uses_capacity_without_spare_embedding_row(self):
        candidate = build(Candidate, "cpu").eval()
        tables = []
        handle = candidate.concept_graph.register_forward_pre_hook(
            lambda _module, args: tables.append(args[-1])
        )
        try:
            with torch.no_grad():
                candidate(timed_batch(torch.device("cpu")))
        finally:
            handle.remove()
        self.assertEqual(len(tables), 1)
        self.assertEqual(candidate.hist_concept_emb.num_embeddings, candidate.n_question + 1)
        self.assertEqual(tables[0].size(0), candidate.n_question)
        self.assertEqual(
            candidate.concept_graph.edge_weights(tables[0]).shape,
            (candidate.n_question - 2, candidate.n_question - 2),
        )

    def test_spare_embedding_row_does_not_change_active_graph_predictions(self):
        candidate = build(Candidate, "cpu").eval()
        activate_graph(candidate.concept_graph)
        batch = timed_batch(torch.device("cpu"))
        with torch.no_grad():
            before = candidate(batch)
            spare = candidate.hist_concept_emb.weight[candidate.n_question]
            spare.copy_(100 * torch.sin(torch.arange(spare.numel()) + 1))
            self.assertTrue(torch.equal(before, candidate(batch)))

    def test_spare_embedding_row_has_zero_gradient_with_active_graph(self):
        candidate = build(Candidate, "cpu").eval()
        activate_graph(candidate.concept_graph)
        candidate(timed_batch(torch.device("cpu"))).sum().backward()
        gradient = candidate.hist_concept_emb.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(gradient[candidate.n_question].eq(0).all()))
        self.assertGreater(float(gradient[2:candidate.n_question].abs().sum()), 0)

    def test_graph_remains_active_without_newton(self):
        batch = timed_batch(torch.device("cpu"))
        candidate = build(Candidate, "cpu", use_prequential_newton=0).eval()
        activate_graph(candidate.concept_graph)
        with torch.no_grad():
            active = candidate(batch)
            candidate.use_concept_graph = False
            self.assertFalse(torch.equal(active, candidate(batch)))

    def test_original_attention_ssm_and_stat_methods_are_not_replaced(self):
        for name in ("_attend", "_stats", "_factorized_token", "_boundary_weights", "_sequences", "_modulate_history"):
            self.assertIs(getattr(Candidate, name), getattr(V28, name))
        candidate, parent = build(Candidate, "cpu"), build(V28, "cpu")
        self.assertIs(type(candidate.ssm), type(parent.ssm))
        self.assertTrue(all(isinstance(block["attn"], torch.nn.MultiheadAttention) for block in candidate.blocks))

    def test_full_model_has_no_state_carry(self):
        candidate = build(Candidate, "cpu").eval()
        activate_graph(candidate.concept_graph)
        batch = timed_batch(torch.device("cpu"))
        changed = {key: value.clone() for key, value in batch.items()}
        changed["rseqs"] = 1 - changed["rseqs"]
        with torch.no_grad():
            before = candidate(batch)
            candidate(changed)
            self.assertTrue(torch.equal(before, candidate(batch)))


if __name__ == "__main__":
    unittest.main()
