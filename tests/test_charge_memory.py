import unittest

import torch

from snnformer.charge_memory import ChargeTransferMemory, ChargeTransferState, _scan_clipped_maps


class ChargeMemoryTest(unittest.TestCase):
    def test_clipped_prefix_reverses_after_saturation_and_resets(self):
        shift = torch.tensor([200, -32, 32, 0, -32], dtype=torch.float32).reshape(1, -1, 1)
        lower = torch.tensor([1, 1, 1, 80, 0], dtype=torch.float32).reshape_as(shift)
        upper = torch.tensor([255, 255, 255, 80, 160], dtype=torch.float32).reshape_as(shift)
        expected = torch.tensor([255, 223, 255, 80, 48], dtype=torch.float32).reshape_as(shift)
        self.assertTrue(torch.equal(_scan_clipped_maps(shift, lower, upper), expected))

    def test_coarse_total_survives_saturating_fine_updates(self):
        modules = [ChargeTransferMemory(1, 2, conserve=conserve) for conserve in (True, False)]
        with torch.no_grad():
            for module in modules:
                module.project.weight.zero_()
                module.project.weight[2, 0] = 1.0
                module.bias[0, :] = -10
                module.bias[[4, 8], :] = 10
        spikes = torch.tensor([16.] * 6 + [-16.]).reshape(1, -1, 1)
        for module in modules:
            stats = {}
            reads = module(spikes, stats=stats).unflatten(-1, (1, 2, 4))
            if module.conserve:
                self.assertTrue(torch.equal(reads[:, :, :, 0], reads[:, :1, :, 0].expand_as(reads[:, :, :, 0])))
                self.assertGreater(reads[:, :, :, 1].squeeze().unique(dim=0).shape[0], 1)
                self.assertEqual(stats["fine_coarse_drift_ticks"].item(), 0)
                # Pure opposite forward drives maximize left, then back away.
                state = None
                for frame in torch.tensor([16.] * 20 + [-16.]):
                    _, state = module.step(frame.reshape(1, 1), state)
                self.assertEqual(state.left.item(), 239)
                self.assertEqual(state.right.item(), 17)
            else:
                self.assertGreater(stats["fine_coarse_drift_ticks"].item(), 0)
            self.assertEqual(stats["probe_events"].item(), stats["read_spikes"].item())

    def test_batch_matches_stream_across_holds_and_clears(self):
        torch.manual_seed(11)
        spikes = torch.randint(-6, 7, (2, 527, 10)).float()
        clear = torch.zeros(spikes.shape[:2], dtype=torch.bool)
        clear[0, [7, 8, 126, 443]] = True
        clear[1, [0, 64, 388]] = True

        for conserve in (True, False):
            with self.subTest(conserve=conserve):
                bank = ChargeTransferMemory(10, 16, conserve=conserve)
                state, expected = None, []
                for frame, reset in zip(spikes.unbind(1), clear.unbind(1)):
                    if state is not None:
                        reset = reset[:, None]
                        state = ChargeTransferState(state.left.masked_fill(reset, 128),
                                                    state.right.masked_fill(reset, 128),
                                                    state.cooldown.masked_fill(reset, 0))
                    read, state = bank.step(frame, state)
                    expected.append(read)
                    self.assertTrue(((0 <= state.left) & (state.left <= 255)).all())
                    self.assertTrue(((0 <= state.right) & (state.right <= 255)).all())
                actual = bank(spikes, clear=clear)
                self.assertTrue(torch.equal(actual, torch.stack(expected, dim=1)))
                self.assertTrue(((actual == 0) | (actual == 1)).all())

    def test_old_coarse_ablation_obeys_read_before_commit_and_clear(self):
        bank = ChargeTransferMemory(1, 2)
        with torch.no_grad():
            bank.project.weight.zero_()
            bank.bias[0, :] = 10
            bank.bias[1, :] = 5
            bank.bias[[4, 8], :] = 10
            bank.bias[5:8, 0] = torch.tensor([.375, .625, .875]).logit()
        spikes = torch.zeros(1, 69, 1)
        clear = torch.zeros(1, 69, dtype=torch.bool)
        clear[:, 67] = True
        stats = {}
        factual = bank(spikes, clear=clear).unflatten(-1, (1, 2, 4))
        ablated = bank(spikes, clear=clear, ablate_min_age=2, stats=stats).unflatten(-1, (1, 2, 4))
        changed_coarse = (factual[..., 0, :] != ablated[..., 0, :]).any(dim=-1).squeeze()
        self.assertTrue(changed_coarse[2:65].all())
        self.assertTrue(changed_coarse[66])
        self.assertFalse(changed_coarse[[0, 1, 65, 67, 68]].any())
        self.assertTrue(torch.equal(factual[..., 1, :], ablated[..., 1, :]))
        self.assertEqual(stats["accepted_write_events"].item(), 3)
        self.assertEqual(stats["old_probe_events"].item(), changed_coarse.sum().item())
        self.assertEqual(stats["altered_probe_events"].item(), changed_coarse.sum().item())

    def test_gradients_reach_coarse_commit_and_fine_drives(self):
        torch.manual_seed(19)
        spikes = torch.randint(-3, 4, (2, 91, 8)).float().requires_grad_()
        for conserve in (True, False):
            bank = ChargeTransferMemory(8, 8, conserve=conserve)
            actual = bank(spikes)
            (actual * torch.randn_like(actual)).mean().backward()
            self.assertTrue(bank.project.weight.grad.isfinite().all())
            self.assertTrue(spikes.grad.isfinite().all())
            self.assertTrue((bank.bias.grad[1:4].abs().sum(dim=-1) > 0).all())


if __name__ == "__main__":
    unittest.main()
