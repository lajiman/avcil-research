"""Controlled sampling, membership-aware CMR, historical guards and CPU smoke test."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import offline_memory_protocols as protocol
import probe_checkpoint_losses as probe
import probe_checkpoint_three_way as three

try:
    probe.load_runtime()
    TORCH_READY = True
except ImportError:
    TORCH_READY = False


def fake_dataset(size=16):
    # Deliberately not lexicographically ordered: training preserves this order.
    classes = {c:[f'{c}_{i}' for i in reversed(range(size))] for c in range(6)}
    return SimpleNamespace(_get_class_vids=lambda c:classes[c], _has_feature=lambda v:True,
        all_id_category_dict={v:str(c) for c, ids in classes.items() for v in ids},
        category_encode_dict={str(c):c for c in classes})


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.data = fake_dataset()
        self.args = SimpleNamespace(seed=42, memory_size=12, class_num_per_step=2)
        self.original = protocol.reconstruct_replay_ids(self.data, self.args, [1,2])

    def test_reconstruction_preserves_global_rng_and_old_prefixes(self):
        before = random.getstate()
        self.assertEqual(self.original, protocol.reconstruct_replay_ids(self.data, self.args, [1,2]))
        self.assertEqual(before, random.getstate())
        for c in [0,1]:
            first = [v for v in self.original['1'] if protocol.sample_label(self.data, v) == c]
            second = [v for v in self.original['2'] if protocol.sample_label(self.data, v) == c]
            self.assertEqual(second, first[:3])

    def test_matched_queries_disjoint_banks_and_reproducibility(self):
        plan = protocol.make_three_way_plan(self.data, self.args, 2, self.original['2'], 7, 3, 4, 8)
        self.assertEqual(plan, protocol.make_three_way_plan(self.data, self.args, 2, self.original['2'], 7, 3, 4, 8))
        protocol.validate_three_way_plan(self.data, self.args, 2, plan, self.original['2'], 3, 4, 8)
        self.assertFalse(set(plan['fresh_memory_ids']) & set(self.original['2']))
        self.assertEqual(protocol.GROUPS['heldout_fixed_reference'][1], protocol.GROUPS['fresh_memory'][1])
        broken = copy.deepcopy(plan)
        broken['fresh_memory_ids'][0] = self.original['2'][0]
        with self.assertRaises(ValueError):
            protocol.validate_three_way_plan(self.data, self.args, 2, broken, self.original['2'], 3, 4, 8)

    def test_no_silent_resizing_or_overlap_when_pools_are_small(self):
        small = fake_dataset(6)
        original = protocol.reconstruct_replay_ids(small, self.args, [1])['1']
        with self.assertRaisesRegex(ValueError, 'non-memory'):
            protocol.make_three_way_plan(small, self.args, 1, original, 7, 1, 4, 4)
        with self.assertRaisesRegex(ValueError, 'batch size'):
            protocol.make_three_way_plan(self.data, self.args, 2, self.original['2'], 7, 1, 4, 99)

    def test_first_maximum_validation_epoch_and_bad_references(self):
        rows = [dict(step='1', epoch=str(i), val_acc=str(acc)) for i, acc in [(2,.8),(0,.5),(1,.8)]]
        self.assertEqual(protocol.best_epoch_row(rows, 1)['epoch'], '1')
        row = dict(step=1, class_id=0, category_name='a', prototype_count=5.,
            reliability_a_from_v=.2, reliability_v_from_a=.3, trust_a_from_v=.15, trust_v_from_a=.2,
            cmr_weight_a_from_v=1., cmr_weight_v_from_a=1.)
        self.assertTrue(all(r['passed'] for r in protocol.compare_reference_rows([row], [row], 1)))
        wrong = dict(row, trust_a_from_v=.25)
        self.assertFalse(all(r['passed'] for r in protocol.compare_reference_rows([wrong], [row], 1)))
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            protocol.indexed_class_rows([row, row], 2, 1)


@unittest.skipUnless(TORCH_READY, 'Requires CPU PyTorch/h5py/numpy/tqdm')
class RuntimeTests(unittest.TestCase):
    def test_reconstruction_matches_actual_dataloader_sequence(self):
        data, args = fake_dataset(), SimpleNamespace(seed=42, memory_size=12, class_num_per_step=2)
        loader = probe.exemplarLoader.__new__(probe.exemplarLoader)
        loader.args = args
        loader.exemplar_class_vids_set = []
        loader._get_class_vids, loader._has_feature = data._get_class_vids, data._has_feature
        loader.close_visual_features_h5 = lambda:None
        before = random.getstate()
        actual = {}
        try:
            random.seed(args.seed)
            for step in [1,2]:
                loader.incremental_step = step
                loader._update_exemplars_()
                actual[str(step)] = list(loader.exemplar_vids_set)
        finally:
            random.setstate(before)
        self.assertEqual(actual, protocol.reconstruct_replay_ids(data, args, [1,2]))

    def test_external_and_mixed_queries_use_correct_prototypes_and_gradients(self):
        torch, F, rd = probe.torch, probe.F, probe.rd
        torch.manual_seed(13)
        teacher_a, teacher_v = (F.normalize(torch.randn(6,5), dim=1) for _ in range(2))
        sums_a, sums_v = teacher_a.reshape(3,2,5).sum(1), teacher_v.reshape(3,2,5).sum(1)
        bank = rd.TeacherPrototypeBank(sums_a, sums_v, F.normalize(sums_a,dim=1),
            F.normalize(sums_v,dim=1), torch.full((3,),2.))
        ca, cv = (F.normalize(torch.randn(2,5), dim=1).requires_grad_() for _ in range(2))
        ta = torch.stack([teacher_a[0], F.normalize(torch.randn(5),dim=0)])
        tv = torch.stack([teacher_v[0], F.normalize(torch.randn(5),dim=0)])
        labels, member = torch.tensor([0,1]), torch.tensor([True,False])
        terms = protocol.membership_margin_terms(ca, cv, ta, tv, labels, bank, .1, .01, member)
        def manual(query, sums, positives):
            margins = []
            for i in range(2):
                prototypes = F.normalize(sums,dim=1).clone()
                if member[i]:
                    prototypes[labels[i]] = F.normalize(sums[labels[i]]-positives[i],dim=0)
                scores = query[i] @ prototypes.T / .1
                margins.append(scores[labels[i]]-torch.logsumexp(scores[torch.arange(3)!=labels[i]],dim=0))
            return torch.stack(margins)
        expected_a, expected_v = manual(ca,sums_v,tv), manual(cv,sums_a,ta)
        torch.testing.assert_close(terms.cur_a_from_v, expected_a)
        torch.testing.assert_close(terms.cur_v_from_a, expected_v)
        for actual, expected in zip(torch.autograd.grad(terms.cur_a_from_v.sum()+terms.cur_v_from_a.sum(), (ca,cv)),
                                    torch.autograd.grad(expected_a.sum()+expected_v.sum(), (ca,cv))):
            torch.testing.assert_close(actual, expected)
        # The old unconditional subtraction would corrupt the external query.
        wrong = rd.compute_margin_terms(ca,cv,ta,tv,labels,bank,.1,.01)
        self.assertGreater(float((wrong.cur_a_from_v[1]-terms.cur_a_from_v[1]).abs()), .01)
        all_members = torch.ones(2,dtype=torch.bool)
        unchanged = protocol.membership_margin_terms(ca,cv,ta,tv,labels,bank,.1,.01,all_members)
        for key in vars(wrong):
            torch.testing.assert_close(getattr(unchanged,key),getattr(wrong,key))

    def test_end_to_end_three_groups_and_failed_identity_guard(self):
        import h5py
        import numpy as np
        from model.audio_visual_model_incremental import IncreAudioVisualNet
        from rd_crosssdc.diagnostics import save_static_bank, save_dynamic_weights
        torch = probe.torch
        torch.set_num_threads(2)
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory(prefix='three-way-probe-test-') as temp:
            root = Path(temp)
            run_dir, feature_root, meta_root = root/'save'/'tiny', root/'features', root/'meta'
            metrics = root/'save'/'metrics'/'tiny'/'rd_crosssdc'
            for p in [run_dir, feature_root/'audio_pretrained_feature', meta_root, metrics]:
                p.mkdir(parents=True)
            classes = {c:[f'c{c}_{i}' for i in reversed(range(12))] for c in range(6)}
            labels = {v:str(c) for c, ids in classes.items() for v in ids}
            np.save(meta_root/'all_id_category_dict.npy', {'train':labels})
            np.save(meta_root/'all_classId_vid_dict.npy', {'train':classes})
            np.save(meta_root/'category_encode_dict.npy', {str(c):c for c in classes})
            rng = np.random.default_rng(17)
            np.save(feature_root/'audio_pretrained_feature'/'audio_pretrained_feature_dict.npy',
                    {v:rng.normal(size=768).astype('float32') for v in labels})
            with h5py.File(feature_root/'visual_features.h5','w') as handle:
                for v in labels:
                    handle.create_dataset(v,data=rng.normal(size=(8,1,768)).astype('float32'))
            args = SimpleNamespace(dataset='VGGSound',modality='audio-visual',num_classes=6,class_num_per_step=2,
                seed=42,memory_size=8,feature_root=str(feature_root),meta_root=str(meta_root),num_workers=0,
                train_batch_size=4,exemplar_batch_size=4,instance_contrastive=True,class_contrastive=True,
                attn_score_distil=True,instance_contrastive_temperature=.1,class_contrastive_temperature=.1,
                cross_sdc_temperature=.1,rd_mode='adaptive_crosssdc_cmr',rd_margin_temperature=.1,
                rd_margin_tolerance=.01,rd_cmr_penalty='hinge',rd_cmr_scale=1.,rd_class_weight_alpha=.5,
                rd_trust_offset=.05,rd_trust_gamma=1.,rd_need_delta=.05,rd_need_eta=0.,rd_need_ema_momentum=.9,
                rd_weight_min=.5,rd_weight_max=2.,rd_trust_shrinkage_beta=10.,lam_I=.1,lam_C=1.,
                lam_cross_sdc_i=.1,lam_cross_sdc_c=0.,lam_cmr=.03,lam=.5)
            for step in [0,1,2]:
                torch.save(IncreAudioVisualNet(args,(step+1)*2),run_dir/f'step_{step}_best_model.pkl')
            dataset = probe.exemplarLoader(args,modality='audio-visual')
            history, true_ids = [], {}
            before_rng = random.getstate()
            try:
                random.seed(args.seed)
                for step in [1,2]:
                    dataset._set_incremental_step_(step)
                    true_ids[str(step)] = list(dataset.exemplar_vids_set)
                    teacher = torch.load(run_dir/f'step_{step-1}_best_model.pkl',weights_only=False).eval()
                    bank, weights = three.build_reference(teacher,dataset,args,step,true_ids[str(step)],torch.device('cpu'))
                    save_static_bank(str(metrics/f'step_{step}_static_trust.csv'),step,bank.counts,
                        bank.reliability_a_from_v,bank.reliability_v_from_a,bank.trust_a_from_v,bank.trust_v_from_a,
                        {c:str(c) for c in range(step*2)})
                    snapshot = {'trust_a_from_v':bank.trust_a_from_v,'trust_v_from_a':bank.trust_v_from_a}
                    for j,d in enumerate(['a_from_v','v_from_a']):
                        snapshot.update({f'need_{d}':torch.zeros(step*2),f'class_weight_{d}':weights[j],f'cmr_weight_{d}':weights[j]})
                    save_dynamic_weights(str(metrics/f'step_{step}_epoch_1_weights.csv'),step,1,snapshot,{c:str(c) for c in range(step*2)})
                    for epoch,acc in [(2,.7),(0,.5),(1,.7)]:
                        history.append(dict(step=step,epoch=epoch,val_acc=acc,rd_mode=args.rd_mode,cmr_penalty='hinge',
                            cmr_scale=1.,cmr_tolerance=.01,cmr=.1,weighted_cmr=.003,cmr_active_a_from_v=.2,cmr_active_v_from_a=.3))
            finally:
                random.setstate(before_rng)
                dataset.close_visual_features_h5()
            probe.write_csv(metrics/'epoch_summary.csv',history)
            base = ['--run-dir',str(run_dir),'--feature-root',str(feature_root),'--meta-root',str(meta_root),
                    '--steps','1','2','--batches','1','--threads','2']
            out = root/'analysis'
            three.main(base+['--output',str(out)])
            metadata = json.loads((out/'metadata.json').read_text())
            self.assertEqual(metadata['status'],'complete')
            self.assertTrue(all(s['parameters_and_buffers_unchanged'] for s in metadata['states']))
            self.assertEqual(json.loads((out/'original_replay_ids.json').read_text()),true_ids)
            batch_rows, samples = protocol.read_csv(out/'cmr_batches.csv'), protocol.read_csv(out/'cmr_samples.csv')
            self.assertEqual(len(batch_rows),6)
            for batch in batch_rows:
                rs = [r for r in samples if r['experiment_group']==batch['experiment_group'] and r['step']==batch['step']]
                self.assertAlmostEqual(sum(float(r['weighted_cmr_contribution']) for r in rs),float(batch['weighted_cmr']),places=6)
                self.assertTrue(all((r['in_reference_memory']=='True') == (batch['experiment_group']!='heldout_fixed_reference') for r in rs))
            components = protocol.read_csv(out/'components.csv')
            for step in ['1','2']:
                for name in ['ce','kd','instance_contrastive','class_contrastive','cross_sdc_i','attn_spatial','attn_temporal']:
                    values = [r for r in components if r['step']==step and r['parameter_scope']=='all' and r['component']==name]
                    fixed = next(r for r in values if r['experiment_group']=='heldout_fixed_reference')
                    fresh = next(r for r in values if r['experiment_group']=='fresh_memory')
                    self.assertEqual(fixed['raw_loss'],fresh['raw_loss'])
                    self.assertEqual(fixed['raw_grad_norm'],fresh['raw_grad_norm'])
            # Reusing the manifest must preserve scalar results and explicitly omit gradients.
            out2 = root/'loss_only'
            three.main(base+['--output',str(out2),'--probe-manifest',str(out/'probe_manifest.json'),'--loss-only'])
            scalar = protocol.read_csv(out2/'components.csv')
            self.assertTrue(all(r['raw_grad_norm']=='' for r in scalar))
            for r in scalar:
                original = next(x for x in components if x['parameter_scope']=='all' and all(x[k]==r[k] for k in ['step','batch','experiment_group','component']))
                self.assertAlmostEqual(float(r['weighted_loss']),float(original['weighted_loss']),places=6)
            # Same-class wrong IDs pass coverage checks but must fail numerical identity validation.
            wrong = copy.deepcopy(true_ids)
            wrong['1'][0] = next(v for v in classes[0] if v not in wrong['1'])
            wrong_path = root/'wrong_ids.json'
            wrong_path.write_text(json.dumps(wrong))
            badout = root/'bad'
            with self.assertRaisesRegex(ValueError,'reference validation failed'):
                three.main(base+['--output',str(badout),'--replay-ids',str(wrong_path)])
            self.assertEqual(json.loads((badout/'metadata.json').read_text())['status'],'failed')
            self.assertFalse((badout/'components.csv').exists())


if __name__ == '__main__':
    unittest.main()
