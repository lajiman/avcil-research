"""Offline probe math, input contract, and parity with actual training formulas."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import probe_checkpoint_losses as probe

try:
    probe.load_runtime()
    TORCH_READY = True
except ImportError:
    TORCH_READY = False


class GeometryTests(unittest.TestCase):
    def test_legacy_probes_reject_persistent_banks_instead_of_rebuilding_a(self):
        probe.require_memory_prototype_policy(SimpleNamespace())
        probe.require_memory_prototype_policy(SimpleNamespace(rd_prototype_policy='memory'))
        for policy in ('pre_shrink', 'historical'):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, 'saved task bank'):
                probe.require_memory_prototype_policy(SimpleNamespace(rd_prototype_policy=policy))

    def test_opposing_gradients_and_cancellation(self):
        r = probe.lambda_geometry(4., 1., -2., 2.)
        self.assertEqual(r['cmr_over_non_cmr'], 1.)
        self.assertEqual(r['total_grad_norm'], 0.)
        self.assertEqual(r['cmr_cosine_with_non_cmr'], -1.)
        self.assertIsNone(r['total_cosine_with_non_cmr'])
        self.assertEqual(r['lambda_for_equal_grad_norm'], 2.)

    def test_orthogonal_gradients(self):
        r = probe.lambda_geometry(9., 16., 0., 1.)
        self.assertEqual(r['total_grad_norm'], 5.)
        self.assertAlmostEqual(r['total_cosine_with_non_cmr'], .6)

    def test_zero_gradients_are_not_given_fake_cosines(self):
        r = probe.lambda_geometry(0., 0., 0., .1)
        self.assertIsNone(r['cmr_over_non_cmr'])
        self.assertIsNone(r['lambda_for_equal_grad_norm'])
        self.assertIsNone(r['cmr_cosine_with_non_cmr'])

    def test_no_training_or_checkpoint_mutation_calls(self):
        tree = ast.parse((ROOT/'probe_checkpoint_losses.py').read_text(encoding='utf-8'))
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('backward', calls)
        self.assertNotIn('step', calls)
        self.assertNotIn('save', calls)


class ProbePlanTests(unittest.TestCase):
    def setUp(self):
        ids = {c:[f'{c}_{i}' for i in range(8)] for c in range(4)}
        self.data = SimpleNamespace(_get_class_vids=lambda c:ids[c], _has_feature=lambda v:True,
            all_id_category_dict={v:c for c,vs in ids.items() for v in vs},
            category_encode_dict={c:c for c in ids})
        self.args = SimpleNamespace(class_num_per_step=2, memory_size=8)

    def test_reproducible_batches_and_correct_old_only_bank(self):
        a = probe.make_probe_plan(self.data, self.args, 1, 42, 3, 4, 4)
        b = probe.make_probe_plan(self.data, self.args, 1, 42, 3, 4, 4)
        self.assertEqual(a, b)
        self.assertEqual(len(a['memory_ids']), 8)
        probe.validate_plan(self.data, self.args, 1, a)

    def test_leave_one_out_requires_replay_membership(self):
        p = probe.make_probe_plan(self.data, self.args, 1, 42, 1, 4, 4)
        p['batches'][0]['replay_ids'][0] = '2_0'
        with self.assertRaisesRegex(ValueError, 'prototype memory'):
            probe.validate_plan(self.data, self.args, 1, p)


@unittest.skipUnless(TORCH_READY, 'Requires PyTorch/h5py in the training environment')
class RuntimeTests(unittest.TestCase):
    def test_gradient_measurement_preserves_model_and_rng(self):
        torch = probe.torch
        model = torch.nn.Linear(2, 1)
        before = probe.state_fingerprint(model)
        rng = torch.random.get_rng_state().clone()
        ce = sum(p.square().sum() for p in model.parameters())
        cmr = -model.weight.sum()
        rows, _, sweep = probe.measure_gradients(model,
            {'ce':(ce, 1.), 'cmr':(cmr, .3), 'off':(None, 1.)}, ce+.3*cmr, [0., .1, .3])
        self.assertEqual(before, probe.state_fingerprint(model))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        r = next(r for r in rows if r['component']=='cmr' and r['parameter_scope']=='all')
        self.assertAlmostEqual(r['weighted_grad_norm'], .3*2**.5)
        s = [r for r in sweep if r['parameter_scope']=='all']
        self.assertAlmostEqual(s[2]['weighted_cmr_grad_norm'], 3*s[1]['weighted_cmr_grad_norm'])

    def test_real_model_losses_match_training_branch(self):
        # Execute the trainer's actual batch formulas on tiny synthetic inputs.
        # This is a parity test, not a reported scientific measurement.
        torch, rd, exact = probe.torch, probe.rd, probe.exact
        from model.audio_visual_model_incremental import IncreAudioVisualNet
        torch.set_num_threads(2)
        args = SimpleNamespace(modality='audio-visual', dataset='VGGSound', class_num_per_step=2,
            instance_contrastive=True, class_contrastive=True, attn_score_distil=True,
            instance_contrastive_temperature=.1, class_contrastive_temperature=.1,
            cross_sdc_temperature=.05, rd_margin_temperature=.1, rd_margin_tolerance=.01,
            rd_cmr_scale=1., lam_I=.1, lam_C=1., lam_cross_sdc_i=.1, lam_cross_sdc_c=0.,
            lam_cmr=.03, lam=.5)
        torch.manual_seed(7)
        model = IncreAudioVisualNet(args, 6).eval()
        teacher = IncreAudioVisualNet(args, 4).eval().requires_grad_(False)
        current = ((torch.randn(2,8,2,768), torch.randn(2,768)), torch.tensor([4,5]))
        replay = ((torch.randn(8,8,2,768), torch.randn(8,768)), torch.tensor([0,0,1,1,2,2,3,3]))
        with torch.no_grad():
            _, a, v = teacher(visual=replay[0][0], audio=replay[0][1], out_feature_before_fusion=True)
        sums_a, sums_v = a.reshape(4,2,-1).sum(1), v.reshape(4,2,-1).sum(1)
        bank = rd.TeacherPrototypeBank(sums_a, sums_v, probe.F.normalize(sums_a,dim=1),
                                       probe.F.normalize(sums_v,dim=1), torch.full((4,),2.))
        weights = (torch.tensor([.5,1.,1.2,1.3]), torch.tensor([1.3,1.2,1.,.5]))
        controller = SimpleNamespace(class_weight_a=weights[0], class_weight_v=weights[1],
            cmr_weight_a=weights[0], cmr_weight_v=weights[1], accumulate=lambda *a:None)
        tree = ast.parse((ROOT/'train_incremental_rd_crosssdc_modular.py').read_text(encoding='utf-8'))
        loop = next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='samples')
        branch = loop.body[0].orelse
        end = next(i for i,n in enumerate(branch) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='cross_i_sum')
        code = compile(ast.Module(body=branch[:end],type_ignores=[]), '<actual-training-batch>', 'exec')
        names = ['ce_loss','cal_contrastive_loss','class_contrastive_loss','cross_sdc_instance_loss',
                 'weighted_cross_sdc_class_loss','cross_sdc_z1_loss']
        cases = [('adaptive_crosssdc_cmr',p,True) for p in ['hinge','direct','exp','log1p']]
        cases += [('crosssdc_cmr','hinge',True),('adaptive_crosssdc_cmr','hinge',False)]
        for mode, penalty, optional in cases:
            args.rd_mode, args.rd_cmr_penalty = mode, penalty
            args.instance_contrastive = args.class_contrastive = args.attn_score_distil = optional
            w = weights if mode=='adaptive_crosssdc_cmr' else (torch.ones(4),)*2
            env = dict(torch=torch, F=probe.F, args=args, step=2, samples=(current,replay), model=model,
                old_model=teacher, device=torch.device('cpu'), last_step_out_class_num=4,
                distillation_temperature=2, prototype_bank=bank, adaptive_controller=controller if mode=='adaptive_crosssdc_cmr' else None,
                uses_cmr=lambda _:True, uses_adaptive_weights=lambda _:mode=='adaptive_crosssdc_cmr',
                uses_persistent_bank=lambda _:False,  # This parity fixture exercises original A.
                compute_margin_terms=rd.compute_margin_terms, cmr_loss=rd.cmr_loss,
                **{n:getattr(exact,n) for n in names})
            exec(code,env)
            components,total,_ = probe.probe_components(model, teacher, current, replay, args, 2, bank, w)
            torch.testing.assert_close(total, env['loss'])
            for n,k in [('ce','loss_CE'),('kd','loss_KD'),('cmr','current_cmr_loss'),
                        ('cross_sdc_i','cross_sdc_inst_loss'),('cross_sdc_c','cross_sdc_cls_loss')]:
                torch.testing.assert_close(components[n][0], env[k])
            g1 = torch.autograd.grad(total, tuple(model.parameters()), allow_unused=True)
            g2 = torch.autograd.grad(env['loss'], tuple(model.parameters()), allow_unused=True)
            for x,y in zip(g1,g2):
                if x is not None:
                    torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-6)
                else:
                    self.assertIsNone(y)


if __name__ == '__main__':
    unittest.main()
