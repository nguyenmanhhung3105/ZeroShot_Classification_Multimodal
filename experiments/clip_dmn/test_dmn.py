import unittest
import numpy as np
from experiments.clip_zlap.core import unit_vectors
from experiments.clip_dmn.core import predict_stream


def fixture(n=29,c=3):
    rng=np.random.default_rng(127)
    return unit_vectors(rng.normal(size=(n,12))),unit_vectors(rng.normal(size=(c,12)))


class DmnTests(unittest.TestCase):
    def test_causal_prefix_and_memory_reset(self):
        x,t=fixture();a,trace,d=predict_stream(x,t,memory_size=3)
        prefix,_,_=predict_stream(x[:11],t,memory_size=3)
        again,_,_=predict_stream(x,t,memory_size=3)
        np.testing.assert_array_equal(a[:11],prefix);np.testing.assert_array_equal(a,again)
        np.testing.assert_allclose(a.sum(1),1,atol=1e-6)
        self.assertEqual(d['selected_views'],1);self.assertLessEqual(max(d['memory_counts']),3)
        self.assertEqual(int(a[0].argmax()),int(trace['pseudo_label'][0]))

    def test_memory_keeps_lowest_entropy_per_pseudo_class(self):
        x,t=fixture(n=71);_,trace,d=predict_stream(x,t,memory_size=3)
        for label,stored in enumerate(d['memory_indices']):
            actual=[i for i in stored if i>=0]
            candidates=np.flatnonzero(trace['pseudo_label']==label)
            expected=candidates[np.argsort(trace['entropy'][candidates])[:3]]
            self.assertEqual(set(actual),set(expected.tolist()))

    def test_true_multiview_features_and_identity_order_coverage(self):
        x,t=fixture();rng=np.random.default_rng(3)
        views=np.stack([x,unit_vectors(rng.normal(size=x.shape)),unit_vectors(rng.normal(size=x.shape))],1)
        p,trace,d=predict_stream(views,t,order=rng.permutation(len(x)),selection_fraction=.67)
        self.assertEqual(p.shape,(len(x),len(t)));self.assertEqual(d['selected_views'],2)
        self.assertFalse(d['single_view_adaptation']);self.assertEqual(len(trace['entropy']),len(x))

    def test_empty_memory_text_anchor_and_rejects_bad_order_or_inputs(self):
        x,t=fixture()
        for kwargs in [{'memory_size':0},{'order':np.zeros(len(x),dtype=int)},{'text_weight':2.}]:
            with self.assertRaises(ValueError):predict_stream(x,t,**kwargs)
        with self.assertRaises(ValueError):predict_stream(x*2,t)

    def test_parent_product_can_resolve_disagreement_in_both_directions(self):
        from experiments.clip_dmn.hierarchy import parent_product
        labels=['not_humanitarian']+[f'class_{i}' for i in range(7)]
        root=np.array([[.45,.55],[.55,.45]])
        leaf=np.zeros((2,8));leaf[0,1]=.05;leaf[1,0]=.05
        pred,binary,scores=parent_product(root,leaf,['informative','not_informative'],labels)
        self.assertEqual(pred,['class_0','not_humanitarian']);self.assertEqual(binary,['informative','not_informative'])
        np.testing.assert_allclose(scores.sum(1),1.)
        swapped=[7,6,5,4,3,2,1,0]
        reordered=parent_product(root[:,::-1],leaf[:,swapped],['not_informative','informative'],[labels[i] for i in swapped])
        self.assertEqual(reordered[0],pred)
        with self.assertRaises(ValueError):parent_product(root*2,leaf,['informative','not_informative'],labels)

    def test_runner_reports_both_tasks_and_keeps_source_order_out_of_mean(self):
        import csv,json,tempfile
        from copy import deepcopy
        from pathlib import Path
        from unittest.mock import patch
        from experiments.clip_dmn import run as runner
        from prompts.crisismmd_prompts import get_label_names
        x,info=fixture(n=17,c=2);_,human=fixture(n=17,c=8)
        il=list(get_label_names('informativeness'));hl=list(get_label_names('humanitarian'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();settings=json.loads((runner.ROOT/'experiments/clip_dmn/config.json').read_text())
            for name in ['majority_run','dn_run','inmap_run']:
                folder=root/name;folder.mkdir();(folder/'run_manifest.json').write_text('{}');settings[name]=str(folder)
            for scope in ['train','dev']:settings[scope+'_run']=str(root/scope)
            settings['text_prototype_run']=settings['train_run']
            def source(path):
                scope=Path(path).name
                return {'directory':str(path),'split':scope,'feature_sha256':'synthetic',
                    'manifest':{'model':'metaclip2_vith14','logit_scale':100.},'features':x,
                    'info_labels':il,'labels':hl,'prototypes':info,
                    'scores':[x@info.T*.95,x@human.T*.9],
                    'rows':[{'id':str(i),'y_true':hl[i%8]} for i in range(len(x))],
                    'binary_truth':[il[i%2] for i in range(len(x))] if scope=='train' else None}
            gates={seed:np.arange(len(x))%2==seed%2 for seed in [0,1,2]}
            with patch.object(runner,'load_features',side_effect=source),patch.object(runner,'text_prototypes',return_value=human), \
                 patch.object(runner,'validate_majority',return_value={}),patch.object(runner,'majority_gates',return_value=gates), \
                 patch.object(runner,'load_companions',return_value=(x@human.T,np.arange(len(x))%2==0)), \
                 patch.object(runner,'read_artifact',return_value='synthetic'):
                output=runner.run(settings,root/'output')
            analysis=json.loads((output/'analysis.json').read_text())
            for scope in ['train','dev']:
                for name,reports in analysis[scope]['by_method'].items():
                    expected=np.mean([r['humanitarian']['macro_f1'] for r in reports if r['order']!='source_order'])
                    self.assertAlmostEqual(expected,analysis[scope]['shuffle_mean'][name]['humanitarian']['macro_f1'])
                    if scope=='dev':self.assertIsNone(analysis[scope]['shuffle_mean'][name]['binary'])
                    else:self.assertIsNotNone(analysis[scope]['shuffle_mean'][name]['binary'])
                with (output/scope/'shuffle_0/predictions.tsv').open() as f:
                    reader=csv.DictReader(f,delimiter='\t');self.assertEqual(len(reader.fieldnames),len(set(reader.fieldnames)))
                    self.assertEqual(len(list(reader)),len(x))
            self.assertEqual(json.loads((output/'run_manifest.json').read_text())['status'],'complete')


if __name__=='__main__':unittest.main()
