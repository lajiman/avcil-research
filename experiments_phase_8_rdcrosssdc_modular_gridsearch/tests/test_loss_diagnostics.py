"""Objective accounting, optional branches, and non-invasive gradient probes."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rd_crosssdc.loss_diagnostics import LossEpochRecorder, gradient_probe_rows

try:
    import torch
except ImportError:
    torch = None


class ScalarAccountingTests(unittest.TestCase):
    def test_negative_direct_and_zero_weight_term(self):
        recorder = LossEpochRecorder()
        recorder.add(1.8, {'ce':(2.,1.), 'cmr':(-2.,.1), 'cross_c':(3.,0.), 'kd':(None,1.)})
        recorder.add(.6, {'ce':(1.,1.), 'cmr':(-4.,.1), 'cross_c':(5.,0.), 'kd':(None,1.)})
        rows = {r['component']:r for r in recorder.rows(step=1,epoch=0,val_acc=.5)}
        self.assertAlmostEqual(rows['cmr']['raw_mean'], -3.)
        self.assertAlmostEqual(rows['cmr']['weighted_mean'], -.3)
        self.assertAlmostEqual(rows['cmr']['signed_weighted_over_total'], -.25)
        self.assertEqual(rows['cross_c']['raw_mean'], 4.)
        self.assertEqual(rows['cross_c']['weighted_mean'], 0.)
        self.assertEqual(rows['kd']['enabled'], 0)
        self.assertIsNone(rows['kd']['raw_mean'])
        self.assertLess(rows['ce']['reconstruction_max_abs_residual'], 1e-12)

    def test_zero_total_has_no_defined_ratio(self):
        recorder = LossEpochRecorder()
        recorder.add(0., {'ce':(1.,1.), 'cmr':(-1.,1.)})
        self.assertIsNone(recorder.rows(step=1,epoch=0,val_acc=0.)[0]['signed_weighted_over_total'])

    def test_accounting_discrepancy_is_visible(self):
        recorder = LossEpochRecorder()
        recorder.add(2., {'ce':(1.,1.)})
        self.assertEqual(recorder.rows(step=0,epoch=0,val_acc=0.)[0]['reconstruction_max_abs_residual'],1.)

    def test_partial_component_does_not_silently_change_denominator(self):
        recorder = LossEpochRecorder()
        recorder.add(1., {'ce':(1.,1.), 'cmr':(None,.1)})
        recorder.add(1.1, {'ce':(1.,1.), 'cmr':(1.,.1)})
        with self.assertRaises(ValueError):
            recorder.rows(step=1,epoch=0,val_acc=0.)

    def test_actual_training_loss_and_logging_branches_reconcile(self):
        # Execute the real scalar assembly and logging AST with numeric inputs.
        # No model or fake autograd is used; tensor semantics are covered below
        # when PyTorch is available.
        source = Path(__file__).resolve().parents[1]/'train_incremental_rd_crosssdc_modular.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        sample_loop = next(n for n in ast.walk(tree) if isinstance(n,ast.For)
                           and isinstance(n.target,ast.Name) and n.target.id=='samples')
        step_branch = sample_loop.body[0]
        addition_start = next(i for i,n in enumerate(step_branch.orelse)
                              if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets))
        # loss=CE+KD, optional I/C, CrossSDC, optional CMR, optional attention.
        objective = step_branch.orelse[addition_start:addition_start+6]
        logging = next(n for n in sample_loop.body if isinstance(n,ast.If)
                       and isinstance(n.test,ast.Name) and n.test.id=='record_components')
        for step in [0,1]:
            for mask in range(16):
                args=SimpleNamespace(instance_contrastive=bool(mask&1),class_contrastive=bool(mask&2),
                    attn_score_distil=bool(mask&4),lam_I=.1,lam_C=1.,lam_cross_sdc_i=.1,
                    lam_cross_sdc_c=0.,lam_cmr=.03,lam=.5,loss_grad_probe_every=0)
                recorder=LossEpochRecorder()
                env=dict(step=step,epoch=0,num_steps=0,loss=2.,loss_CE=2.,loss_KD=.2,
                    instance_contra_loss=4.,class_contra_loss=1.,cross_sdc_inst_loss=3.,
                    cross_sdc_cls_loss=2.,current_cmr_loss=-.1,spatial_attn_dist_loss=.6,
                    temporal_attn_dist_loss=.4,args=args,uses_cmr=lambda a:bool(mask&8),
                    record_components=True,loss_recorder=recorder)
                nodes=([*objective] if step else [])+[logging]
                exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
                rows=recorder.rows(step=step,epoch=0,val_acc=.5)
                self.assertEqual(len(rows),9)
                self.assertLess(rows[0]['reconstruction_max_abs_residual'],1e-12)
                if step==0:
                    self.assertEqual(sum(r['enabled'] for r in rows),1)


@unittest.skipIf(torch is None, 'PyTorch unavailable: autograd checks require the training environment')
class GradientProbeTests(unittest.TestCase):
    def test_probe_preserves_rng_grads_and_optimizer_update(self):
        p=torch.nn.Parameter(torch.tensor([1.,-2.]))
        unused=torch.nn.Parameter(torch.tensor([4.]))
        before=p.detach().clone()
        p.grad=torch.tensor([7.,8.])
        ce=(p**2).sum()
        cmr=-p[0]
        total=ce+.3*cmr
        rng=torch.random.get_rng_state().clone()
        rows=gradient_probe_rows(total,{'ce':(ce,1.),'cmr':(cmr,.3),'off':(None,1.)},
                                 [p,unused],step=1,epoch=0,batch=0)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(p.grad,torch.tensor([7.,8.])))
        self.assertTrue(torch.equal(p.detach(),before))
        cmr_row=next(r for r in rows if r['component']=='cmr')
        self.assertAlmostEqual(cmr_row['weighted_grad_norm'],.3)
        self.assertAlmostEqual(cmr_row['cosine_with_non_cmr'],-1/(5**.5))
        optimizer=torch.optim.Adam([p,unused],lr=.01,weight_decay=.01)
        optimizer.zero_grad(); total.backward(); optimizer.step()
        control=torch.nn.Parameter(before.clone())
        control_opt=torch.optim.Adam([control],lr=.01,weight_decay=.01)
        control_opt.zero_grad()
        ((control**2).sum()-.3*control[0]).backward(); control_opt.step()
        torch.testing.assert_close(p,control,rtol=0,atol=0)

    def test_zero_gradients_and_zero_coefficients(self):
        p=torch.nn.Parameter(torch.tensor([0.]))
        zero=(p**2).sum()
        cmr=p.sum()
        rows=gradient_probe_rows(zero,{'ce':(zero,1.),'cmr':(cmr,0.)},[p],step=0,epoch=0,batch=0)
        for row in rows:
            self.assertIsNone(row['cosine_with_non_cmr'])
            self.assertIsNone(row['weighted_grad_over_non_cmr'])
        self.assertEqual(rows[1]['weighted_grad_norm'],0.)


if __name__ == '__main__':
    unittest.main()
