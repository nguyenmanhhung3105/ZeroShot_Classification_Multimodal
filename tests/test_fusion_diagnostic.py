"""Chẩn đoán bằng ảnh/vector tổng hợp và ViT nhỏ khởi tạo ngẫu nhiên; không tải weights."""
import contextlib
import copy
import csv
import io as stdio
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.special import softmax
import torch
import torch.nn.functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_project import calip, diagnostic, evaluate, io, kaggle, protocol, single_stage


def unit(x):
    return x/np.linalg.norm(x,axis=-1,keepdims=True)


class TinyModel(torch.nn.Module):
    def __init__(self,after_pool=False):
        super().__init__()
        from open_clip.transformer import VisionTransformer
        torch.manual_seed(129)
        self.visual=VisionTransformer(image_size=16,patch_size=4,width=16,layers=1,
                                     heads=2,mlp_ratio=2,output_dim=8,final_ln_after_pool=after_pool)
        self.eval()

    def encode_image(self,x):
        return self.visual(x)


class TinyBackend:
    def __init__(self,device='cpu',after_pool=False):
        self.device='cpu';self.model=TinyModel(after_pool)
        self.metadata={'synthetic':True,'dtype':'torch.float32'}
        self.unloads=0

    def preprocess(self,image):
        return torch.from_numpy(np.asarray(image.resize((16,16))).copy()).permute(2,0,1).float()/255

    def prototypes(self,bank):
        return unit(np.random.default_rng(99).normal(size=(len(bank),8))).astype(np.float32)

    def unload(self):
        self.unloads+=1


class FusionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        old=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,old)

    def make_data(self):
        root=self.root/'toy';(root/'images').mkdir(parents=True)
        io.write_json(root/'classes.json',['a','b','c'])
        io.write_json(root/'provenance.json',{'dataset':'toy','split':'test'})
        rows=[]
        for index in range(8):
            image=root/'images'/f'{index}.png'
            pixels=np.random.default_rng(index).integers(0,256,(16,16,3),dtype=np.uint8)
            Image.fromarray(pixels).save(image)
            rows.append({'id':str(index),'image':image.name,'label':index%3,
                         'bytes':image.stat().st_size,'sha256':io.sha256(image)})
        evaluate.table(root/'manifest.tsv',rows)
        return root,rows

    def test_calip_matches_independent_formula_and_permutations(self):
        rng=np.random.default_rng(4)
        g=unit(rng.normal(size=(2,8)));p=unit(rng.normal(size=(2,6,8)));t=unit(rng.normal(size=(5,8)))
        actual,terms=calip.score(g,p,t)
        expected=[]
        for global_vector,spatial in zip(g,p):
            attention=2*spatial@t.T
            attended_text=spatial.T@softmax(attention,axis=0)
            attended_visual=softmax(attention,axis=1)@t
            pooled=attended_visual.mean(0)+attended_visual.max(0)
            expected.append(100*global_vector@t.T+1.12*100*global_vector@attended_text+.02*100*pooled@t.T)
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-6)
        np.testing.assert_allclose(calip.score(g,p[:,::-1].copy(),t)[0],actual,atol=2e-5,rtol=2e-6)
        np.testing.assert_allclose(calip.score(g,p,t[::-1].copy())[0],actual[:,torch.arange(4,-1,-1)],atol=2e-5,rtol=2e-6)
        self.assertEqual(terms.shape,(2,3,5))

    def test_calip_rejects_wrong_feature_space_and_nonunit_vectors(self):
        with self.assertRaises(ValueError):calip.score(np.ones((2,8)),np.ones((2,4,8)),np.ones((3,8)))
        with self.assertRaises(ValueError):calip.score(np.eye(2),np.ones((2,4,8)),np.eye(2))

    def test_patch_extractor_global_parity_projection_and_single_forward(self):
        for after_pool in [False,True]:
            backend=TinyBackend(after_pool=after_pool)
            image=Image.new('RGB',(16,16),(40,100,200))
            with patch.object(backend.model.visual,'forward',wraps=backend.model.visual.forward) as forward:
                g,p,meta=calip.encode(backend,[image],verify_global=True)
                self.assertEqual(forward.call_count,2)
            with torch.no_grad():
                visual=backend.model.visual;visual.output_tokens=True
                _,tokens=visual(backend.preprocess(image).unsqueeze(0));visual.output_tokens=False
                if after_pool:tokens=visual.ln_post(tokens)
                expected=F.normalize((tokens@visual.proj).float(),dim=-1)
            torch.testing.assert_close(p,expected)
            self.assertEqual(p.shape,(1,16,8));self.assertEqual(g.shape,(1,8))
            self.assertTrue(meta['output_tokens_restored'])
            self.assertEqual(meta['global_reference_max_abs_error'],0.)
            with patch.object(backend.model.visual,'forward',wraps=backend.model.visual.forward) as forward:
                calip.encode(backend,[image])
                self.assertEqual(forward.call_count,1)
            image.close()

    def test_patch_extractor_restores_flag_on_error(self):
        backend=TinyBackend();image=Image.new('RGB',(16,16))
        with patch.object(backend.model.visual,'forward',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):calip.encode(backend,[image])
        self.assertFalse(backend.model.visual.output_tokens)
        image.close()

    def test_subset_random_without_labels_only_opens_selected_images(self):
        root,rows=self.make_data()
        selected=set(np.random.default_rng(0).choice(8,5,replace=False))
        for index in set(range(8))-selected:(root/'images'/rows[index]['image']).unlink()
        with patch.dict(diagnostic.BENCHMARKS,{'toy':(3,8,'test')}):
            names,chosen,cohort=diagnostic.subset(root,'toy',5,0)
            self.assertEqual({int(r['id']) for r in chosen},selected)
            for row in rows:row['label']=(row['label']+1)%3
            evaluate.table(root/'manifest.tsv',rows)
            _,again,_=diagnostic.subset(root,'toy',5,0)
            self.assertEqual([r['id'] for r in chosen],[r['id'] for r in again])
            (root/'images'/chosen[0]['image'].split('/')[-1]).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'SHA256'):diagnostic.subset(root,'toy',5,0)
        self.assertEqual(cohort['full_split_size'],8);self.assertEqual(names,['a','b','c'])

    def test_fusion_validates_posteriors_and_fixed_weights(self):
        branches={'clip':np.array([[.9,.1],[.4,.6]]),'stata':np.array([[.2,.8],[.8,.2]])}
        np.testing.assert_allclose(diagnostic.mix(branches,{'clip':.5,'stata':.5}),[[.55,.45],[.6,.4]])
        for weights in [{'clip':2.},{'clip':1.1,'stata':-.1},{'absent':1.}]:
            with self.assertRaises(ValueError):diagnostic.mix(branches,weights)
        with self.assertRaises(ValueError):diagnostic.probability([[1.,1.]])
        with self.assertRaises(ValueError):diagnostic.probability([[float('nan'),0.]])
        with self.assertRaises(ValueError):diagnostic.mix({**branches,'stata':np.array([[.2,.8]])},{'clip':.5,'stata':.5})
        np.testing.assert_allclose(diagnostic.probability([[.9,.1]],2.),[[.75,.25]])

    def test_single_stage_binary_multiclass_and_class_permutation(self):
        for count in [2, 7]:
            rng=np.random.default_rng(8)
            branches=[softmax(rng.normal(size=(9,count)),axis=1) for _ in range(3)]
            actual=single_stage.candidates(*branches)
            permuted=single_stage.candidates(*(p[:,::-1] for p in branches))
            self.assertEqual(len(actual),11)
            for method,p in actual.items():
                np.testing.assert_allclose(p.sum(1),1.)
                np.testing.assert_allclose(permuted[method],p[:,::-1])
            np.testing.assert_allclose(actual['clip_stata_w0.1'],.9*branches[0]+.1*branches[1])

    def test_stata_grouping_restores_order_tail_and_cpu_threads(self):
        spec=diagnostic.specification();spec.update(adaptation_batch_size=4,cpu_threads=1)
        features=unit(np.random.default_rng(2).normal(size=(9,8)))
        prototypes=unit(np.random.default_rng(3).normal(size=(3,8)))
        initial=softmax(100*features@prototypes.T,axis=1)
        expected=softmax(features@prototypes.T,axis=1)
        old=torch.get_num_threads()
        with patch.object(diagnostic,'stata',side_effect=lambda x,t:(None,softmax(x@t.T,axis=1))) as call,contextlib.redirect_stdout(stdio.StringIO()):
            actual,groups=diagnostic.infer_groups(features,prototypes,initial,spec)
        self.assertEqual(call.call_count,2)
        np.testing.assert_allclose(actual[groups<2],expected[groups<2])
        np.testing.assert_allclose(actual[groups==2],initial[groups==2])
        self.assertEqual(torch.get_num_threads(),old)
        with patch.object(diagnostic,'stata',side_effect=RuntimeError('stop')),contextlib.redirect_stdout(stdio.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'stop'):diagnostic.infer_groups(features,prototypes,initial,spec)
        self.assertEqual(torch.get_num_threads(),old)

    def test_runner_real_tiny_encoder_artifacts_replay_and_label_independence(self):
        root,rows=self.make_data();spec=diagnostic.specification()
        spec.update(samples=8,adaptation_batch_size=4,image_batch_size=4)
        settings=io.config();settings.update(device='cpu',output_root=str(self.root/'results'))
        backends=[]
        def factory(device):
            backend=TinyBackend(device);backends.append(backend);return backend
        with patch.dict(diagnostic.BENCHMARKS,{'toy':(3,8,'test')}),\
             patch.object(diagnostic,'specification',return_value=spec),\
             patch.object(diagnostic,'verify_sources',return_value={'synthetic':True}),\
             contextlib.redirect_stdout(stdio.StringIO()):
            output=diagnostic.run(settings,'toy',root,backend_factory=factory)
            run=protocol.completed_run(output)
            self.assertTrue((output/'single_stage_candidates.tsv').is_file())
            for row in rows:row['label']=(row['label']+1)%3
            evaluate.table(root/'manifest.tsv',rows)
            other=diagnostic.run(settings,'toy',root,backend_factory=factory)
        self.assertEqual([b.unloads for b in backends],[1,1])
        self.assertEqual(run['mode'],'diagnostic');self.assertEqual(run['samples'],8)
        self.assertEqual(len(run['summary']),6)
        self.assertEqual(run['specification']['calibration'],'unfitted_diagnostic_defaults')
        with np.load(output/'replay_scores.npz',allow_pickle=False) as replay,np.load(other/'replay_scores.npz',allow_pickle=False) as relabeled:
            np.testing.assert_allclose(replay['p_fusion_three'],(replay['p_clip']+replay['p_stata']+replay['p_calip'])/3,atol=1e-7)
            for summary in run['summary']:
                p=replay['p_'+summary['method']]
                np.testing.assert_allclose(p,relabeled['p_'+summary['method']],atol=1e-7)
                metrics=evaluate.metric(replay['labels'],p.argmax(1),list(range(3)),p)
                self.assertAlmostEqual(summary['accuracy_mean'],metrics['accuracy'])
                self.assertAlmostEqual(summary['top5_mean'],metrics['top5'])
        with (output/'predictions.tsv').open() as stream:self.assertEqual(len(list(csv.DictReader(stream,delimiter='\t'))),48)
        self.assertTrue((output/'patch_verification.npz').is_file())

    def test_launcher_one_pass_diagnostic_limits_and_private_existing_data(self):
        launcher=yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        original=copy.deepcopy(launcher)
        # Réel packaging de code, staging temporaire; toute API est interdite ici.
        real_path=kaggle.path
        def stage_path(value):
            return self.root/'stage' if str(value)=='results/_system/launches/final' else real_path(value)
        with patch.object(kaggle,'path',side_effect=stage_path),patch.object(kaggle,'cli',side_effect=AssertionError('no API')):
            for limit,mode,samples in [(None,'diagnostic',256),(5,'smoke',5)]:
                plan=kaggle.prepare(launcher,'diagnostic_imagenetv2_fusion',limit)
                document=io.read_json(Path(plan['root'])/'kernel/run.ipynb')
                code='\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type']=='code')
                for c in document['cells']:
                    if c['cell_type']=='code':compile(''.join(c['source']),'<diagnostic notebook>','exec')
                self.assertEqual(code.count('output = run('),1)
                self.assertNotIn('smoke_output',code);self.assertNotIn('smoke=False',code)
                self.assertIn(f'limit={samples}',code)
                self.assertEqual(plan['samples'],samples);self.assertEqual(plan['mode'],mode)
                self.assertEqual(plan['launcher']['timeout_seconds'],3600)
                self.assertIn('imagenetv2-fusion-diagnostic',plan['kernel'])
                meta=io.read_json(Path(plan['root'])/'kernel/kernel-metadata.json')
                self.assertEqual(meta['is_private'],'true')
                self.assertIn('nguyenhung3105/vlm-public-benchmarks-eval-v1',meta['dataset_sources'])
            with self.assertRaises(ValueError):kaggle.prepare(launcher,'diagnostic_imagenetv2_fusion',10000)
        self.assertEqual(launcher,original)

    def test_fakeddit_diagnostic_dev_only_notebook_and_binary_runner(self):
        settings=io.config();settings.update(device='cpu',output_root=str(self.root/'results'))
        rows=[{'id':str(i),'image':'unused','2way':str(i%2),'text':'unused'} for i in range(5)]
        features=unit(np.random.default_rng(4).normal(size=(5,8))).astype(np.float32)
        with patch('vlm_project.data.fakeddit',return_value=(rows,{'split':'dev'})) as loader,\
             patch.object(diagnostic,'verify_sources',return_value={}),\
             patch.object(diagnostic,'extract',return_value=(features,np.ones((5,2)),[])),\
             contextlib.redirect_stdout(stdio.StringIO()):
            output=diagnostic.run(settings,'fakeddit_2way',self.root,limit=5,backend_factory=TinyBackend)
        self.assertEqual(loader.call_args.args[1],'dev')
        self.assertEqual(len(loader.call_args.kwargs['selected_indices']),5)
        run=protocol.completed_run(output)
        self.assertEqual(run['classes'],['fake','real'])
        self.assertTrue(all(r['task']=='fakeddit_2way' for r in run['summary']))
        launcher=yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        real_path=kaggle.path
        with patch.object(kaggle,'path',side_effect=lambda p:self.root/'stage' if str(p)=='results/_system/launches/final' else real_path(p)),\
             patch.object(kaggle,'cli',side_effect=AssertionError('no API')):
            plan=kaggle.prepare(launcher,'diagnostic_fakeddit_2way_fusion',5)
        document=io.read_json(Path(plan['root'])/'kernel/run.ipynb')
        code='\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type']=='code')
        self.assertIn('locate_data("fakeddit", "dev"',code)
        self.assertNotIn('freeze(settings',code)
        self.assertEqual(plan['data_dataset'],'nguyenhung3105/vlm-fakeddit-dev-v2')
        self.assertNotIn('_',plan['kernel'])


if __name__=='__main__':unittest.main()
