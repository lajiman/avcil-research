"""Pairing, candidate geometry, leakage, retention and real inference checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent))
import class_analysis_common as common
import class_geometry_math as geom
import analyze_class_benefits as benefit
import probe_class_geometry as probe


class BenefitTests(unittest.TestCase):
    def test_initial_and_retention_are_not_conflated(self):
        def row(f1):
            return dict(f1=f1,precision=f1,recall=f1,tp=1,fp=0,fn=0,support=1)
        tables = {('avcil',42):{(0,0):row(.8),(1,0):row(.4)},
                  ('hinge_A',42):{(0,0):row(.7),(1,0):row(.5)}}
        rows = common.paired_deltas(tables,['volleyball'],{'volleyball':13})
        final = rows[-1]
        self.assertAlmostEqual(final['delta_f1_pp'],10)
        self.assertAlmostEqual(final['initial_delta_f1_pp'],-10)
        self.assertAlmostEqual(final['relative_retention_delta_f1_pp'],20)
        self.assertAlmostEqual(final['step_delta_change_pp'],20)
        self.assertEqual(final['difficulty_rank'],13)
        self.assertIsNone(rows[0]['step_delta_change_pp'])

    def test_seed_aggregation_preserves_paired_units(self):
        rows = []
        for seed,delta in [(42,1.),(43,-1.),(44,3.)]:
            for c in range(10):
                rows.append(dict(method='A',seed=seed,step=9,class_id=c,difficulty_quartile=1,
                    age_group='6+',first_seen_step=0,difficulty_rank=c,delta_f1_pp=delta,
                    delta_precision_pp=delta,delta_recall_pp=delta,initial_delta_f1_pp=0.,
                    relative_retention_delta_f1_pp=delta))
        per_seed, summary = benefit.grouped_summaries(rows)
        row = next(r for r in summary if r['group_by']=='all')
        self.assertEqual(row['delta_f1_pp_n'],3)
        self.assertEqual(row['delta_f1_pp_mean'],1.)
        self.assertEqual(row['delta_f1_pp_sd'],2.)

    def test_resume_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            common.prepare_output(tmp,{'seed':42})
            common.prepare_output(tmp,{'seed':42},True)
            with self.assertRaises(ValueError):
                common.prepare_output(tmp,{'seed':43},True)
            with self.assertRaises(ValueError):
                common.prepare_output(tmp,{'seed':42})


class GeometryTests(unittest.TestCase):
    def test_separation_and_orthogonal_coordinate_invariance(self):
        ref = np.repeat(np.eye(3),2,axis=0)
        ref_labels = np.repeat(np.arange(3),2)
        query = np.eye(3); labels = np.arange(3)
        rows, scores, _ = geom.geometry(ref,ref_labels,query,labels,3,k=2)
        for r in rows:
            self.assertEqual(r['centroid_accuracy'],1.)
            self.assertEqual(r['knn_purity'],1.)
            self.assertEqual(r['mean_margin'],1.)
        rotation,_ = np.linalg.qr(np.random.default_rng(4).normal(size=(3,3)))
        rotated,_,_ = geom.geometry(ref@rotation,ref_labels,query@rotation,labels,3,k=2)
        np.testing.assert_allclose([r['mean_margin'] for r in rows],[r['mean_margin'] for r in rotated],atol=1e-12)

    def test_logodds_and_retention_known_answer(self):
        ref = np.eye(2); labels = np.arange(2)
        scores = geom.margin_scores(ref,ref,labels,.5)
        np.testing.assert_allclose(scores['logodds_margin'],[2.,2.])
        same = geom.retention(ref,ref,ref,labels,labels,2,.5,.01)
        np.testing.assert_array_equal(same['violation'],[0.,0.])
        reversed_query = ref[::-1]
        bad = geom.retention(ref,reversed_query,ref,labels,labels,2,.5,.01)
        np.testing.assert_allclose(bad['margin_drop'],[4.,4.])
        np.testing.assert_allclose(bad['violation'],[3.99,3.99])

    def test_class_collapse_not_mistaken_for_good_separation(self):
        ref = np.ones((4,3)); labels = np.array([0,0,1,1])
        rows,_,_ = geom.geometry(ref,labels,np.ones((2,3)),np.array([0,1]),2,2)
        np.testing.assert_allclose([r['reference_dispersion'] for r in rows],0.,atol=1e-12)
        np.testing.assert_allclose([r['mean_margin'] for r in rows],0.,atol=1e-12)
        self.assertEqual(sum(r['centroid_accuracy'] for r in rows)/2,.5)

    def test_confusion_reconstructs_counts(self):
        rows,matrix = geom.confusion_rows(np.array([0,0,1,1]),np.array([0,1,1,1]),2)
        np.testing.assert_array_equal(matrix,[[1,1],[0,2]])
        self.assertEqual(rows[0]['recall'],.5)
        self.assertAlmostEqual(rows[1]['f1'],.8)

    def test_full_state_analysis_includes_old_to_new_errors(self):
        probe.np,probe.geom = np,geom
        def data(n):
            result = dict(reference_ids=np.array([f'r{c}' for c in range(n)]),
                query_ids=np.array([f'q{c}' for c in range(n)]),reference_labels=np.arange(n),query_labels=np.arange(n))
            for split in ['reference','query']:
                for feature in ['audio','visual','fusion']:
                    result[split+'_'+feature] = np.eye(20)[:n]
            result['query_logits'] = np.eye(n)
            return result
        current,teacher = data(20),data(10)
        current['query_logits'][0,10]=2.
        order = [f'c{c}' for c in range(100)]; ranks = {name:c for c,name in enumerate(order)}
        historical = {(1,c):dict(f1=1.,tp=1,fp=0,fn=0,support=1) for c in range(20)}
        args = argparse.Namespace(knn_k=1,margin_temperature=.1,retention_tolerance=.01,query_per_class=0)
        rows,matrix = probe.analyze_state(current,teacher,dict(method='avcil',seed=42),1,order,ranks,
            {('avcil',42):historical},args)
        self.assertEqual(len(rows['geometry']),60)
        self.assertEqual(len(rows['retention']),20)
        r = next(r for r in rows['performance'] if r['class_id']==0)
        self.assertEqual(r['old_to_new_error_rate'],1.)
        self.assertEqual(r['old_head_only_recall'],1.)
        self.assertFalse(r['historical_counts_match'])
        teacher['query_ids'][0]='different'
        with self.assertRaises(ValueError):
            probe.analyze_state(current,teacher,dict(method='avcil',seed=42),1,order,ranks,{('avcil',42):historical},args)


class SamplingTests(unittest.TestCase):
    def test_manifest_is_fixed_disjoint_and_checks_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); order=[f'c{c}' for c in range(100)]
            names={name:c for c,name in enumerate(order)}
            ids,classes={},{}
            for split in ['train','test']:
                ids[split]={};classes[split]={}
                for c in range(100):
                    samples=[f'{split}-{c}-{i}' for i in range(3)]
                    classes[split][str(c)]=samples
                    ids[split].update({v:order[c] for v in samples})
            np.save(root/'category_encode_dict.npy',names)
            np.save(root/'all_classId_vid_dict.npy',classes)
            np.save(root/'all_id_category_dict.npy',ids)
            args=argparse.Namespace(meta_root=root,sample_seed=42,reference_per_class=2,query_per_class=0)
            a=probe.make_plan(args,order);b=probe.make_plan(args,order)
            self.assertEqual(a,b)
            self.assertEqual(len(a['reference_ids']),200)
            self.assertEqual(len(a['query_ids']),300)
            self.assertFalse(set(a['reference_ids'])&set(a['query_ids']))
            ids['test']['test-0-0']='c1'
            np.save(root/'all_id_category_dict.npy',ids)
            with self.assertRaises(ValueError):
                probe.make_plan(args,order)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'PyTorch required for real model forward')
class RealForwardTests(unittest.TestCase):
    def test_analysis_forward_preserves_logits_parameters_and_buffers(self):
        import torch
        from model.audio_visual_model_incremental import IncreAudioVisualNet
        args=argparse.Namespace(modality='audio-visual')
        torch.manual_seed(7)
        model=IncreAudioVisualNet(args,20).eval()
        visual=torch.randn(2,8,2,768);audio=torch.randn(2,768)
        probe.torch=torch
        before=probe.checkpoint_hash(model)
        with torch.inference_mode():
            original=model(visual=visual,audio=audio)
            detailed=model(visual=visual,audio=audio,return_dict=True,out_analysis_features=True)
        torch.testing.assert_close(original,detailed['logits'],rtol=0,atol=0)
        self.assertEqual(before,probe.checkpoint_hash(model))
        for key in ['z1_audio_norm','z1_visual_norm','z2_fusion_norm']:
            torch.testing.assert_close(detailed[key].norm(dim=1),torch.ones(2))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    @unittest.skipUnless(importlib.util.find_spec('h5py') is not None,'h5py required for end-to-end fixture')
    def test_two_process_pipeline_and_cache_resume(self):
        import torch
        import h5py
        from model.audio_visual_model_incremental import IncreAudioVisualNet
        with tempfile.TemporaryDirectory(prefix='avcil-geometry-test-', dir=common.native(tempfile.gettempdir())) as tmp:
            root=common.native(tmp); meta=root/'meta'; features=root/'features'
            meta.mkdir(); (features/'audio_pretrained_feature').mkdir(parents=True)
            ranks=common.difficulty(); order=list(ranks); ids={}; classes={}; audio={}
            rng=np.random.default_rng(2)
            with h5py.File(features/'visual_features.h5','w') as store:
                for split in ['train','test']:
                    ids[split]={}; classes[split]={}
                    for c,name in enumerate(order):
                        vid=f'{split}-{c}'
                        ids[split][vid]=name; classes[split][str(c)]=[vid]
                        audio[vid]=rng.normal(size=768).astype(np.float32)
                        store[vid]=rng.normal(size=(8,2,768)).astype(np.float32)
            np.save(features/'audio_pretrained_feature/audio_pretrained_feature_dict.npy',audio)
            np.save(meta/'category_encode_dict.npy',{name:c for c,name in enumerate(order)})
            np.save(meta/'all_classId_vid_dict.npy',classes); np.save(meta/'all_id_category_dict.npy',ids)
            # Create only the selected evaluation/teacher checkpoints. Historical
            # metrics still cover all steps, like the real archive.
            fixtures=[('avcil','save_supp_ckpt_v1','VGGSound_random_balance_avcil_only_supp_ckpt_v1_h200_seed42',0.,0.,.5),
                ('hinge_A','save_commands_cmr_hinge_tolerance_focus_3seeds','VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_tol_focus_v1_hinge_A_lc0p03_a0p5_tol0p01_s1p0_h200_seed42',.1,.03,.5),
                ('hinge_B','save_commands_cmr_hinge_tolerance_focus_3seeds','VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_tol_focus_v1_hinge_B_lc0p1_a1p0_tol0p01_s1p0_h200_seed42',.1,.1,1.)]
            for method,archive,name,cross,lam,alpha in fixtures:
                checkpoint=root/archive/name;checkpoint.mkdir(parents=True)
                metrics=root/archive/'metrics'/name
                rows=[]
                for t in range(10):
                    for c in range((t+1)*10):
                        rows.append(dict(step=t,class_id=c,category_name=order[c],first_seen_step=c//10,
                            support=1,tp=1,fp=0,fn=0,f1=1.,precision=1.,recall=1.))
                common.write_csv(metrics/'per_class_metrics.csv',rows)
                if method!='avcil':
                    for t in range(1,10):
                        static=[dict(class_id=c,prototype_count=5.,trust_a_from_v=.5,trust_v_from_a=.5,
                                     reliability_a_from_v=.5,reliability_v_from_a=.5) for c in range(t*10)]
                        weights=[dict(class_id=c,cmr_weight_a_from_v=1.,cmr_weight_v_from_a=1.) for c in range(t*10)]
                        common.write_csv(metrics/f'rd_crosssdc/step_{t}_static_trust.csv',static)
                        common.write_csv(metrics/f'rd_crosssdc/step_{t}_epoch_0_weights.csv',weights)
                args=argparse.Namespace(modality='audio-visual',seed=42,lam_cross_sdc_i=cross,lam_cmr=lam,
                    rd_class_weight_alpha=alpha,lam_cross_sdc_c=0.,class_num_per_step=10,num_classes=100,
                    rd_cmr_penalty='hinge',rd_margin_tolerance=.01,lam_I=.1,lam_C=1.,lam=.5,memory_size=500,
                    instance_contrastive=True,class_contrastive=True,attn_score_distil=True)
                for t in [0,1]:
                    torch.save(IncreAudioVisualNet(args,(t+1)*10),checkpoint/f'step_{t}_best_model.pkl')
            output=root/'output'
            command=[sys.executable,str(HERE/'run_class_benefit_geometry.py'),'--archive-root',str(root),
                '--feature-root',str(features),'--meta-root',str(meta),'--output',str(output),
                '--metrics-seeds','42','--geometry-seeds','42','--steps','1','--reference-per-class','1',
                '--query-per-class','1','--batch-size','8','--threads','2']
            first=subprocess.run(command,capture_output=True,text=True,timeout=120)
            log='\n'.join(p.read_text(encoding='utf-8') for p in output.glob('*.log'))
            self.assertEqual(first.returncode,0,first.stdout+first.stderr+log)
            self.assertEqual(json.loads((output/'status.json').read_text()),{'metrics':0,'geometry':0})
            geo=output/'geometry'
            comparisons=common.read_csv(geo/'geometry_deltas.csv')
            self.assertEqual(len(comparisons),120)
            self.assertEqual(len(common.read_csv(geo/'retention.csv')),60)
            cache_times={p.name:p.stat().st_mtime_ns for p in (geo/'cache').glob('*.npz')}
            self.assertEqual(len(cache_times),7)
            second=subprocess.run(command+['--resume'],capture_output=True,text=True,timeout=120)
            log='\n'.join(p.read_text(encoding='utf-8') for p in output.glob('*.log'))
            self.assertEqual(second.returncode,0,second.stdout+second.stderr+log)
            self.assertEqual(cache_times,{p.name:p.stat().st_mtime_ns for p in (geo/'cache').glob('*.npz')})


if __name__=='__main__':
    unittest.main()
