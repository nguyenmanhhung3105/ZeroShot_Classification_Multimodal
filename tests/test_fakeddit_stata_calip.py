import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import softmax

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests'))
from vlm_project import fakeddit_calip_ablation as c, kaggle
from test_fakeddit_hierarchy import FakeBackend,binary_prototypes,stata
from vlm_project.io import config


class StatACalipTests(unittest.TestCase):
    def test_extractor_tiny_real_vit_and_patch_replay(self):
        from test_fusion_diagnostic import TinyBackend,unit
        from PIL import Image
        import torch
        backend=TinyBackend()
        backend.encode_texts=lambda texts:unit(np.random.default_rng(6).normal(size=(len(texts),8))).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i in range(5):
                p=root/f'{i}.png';Image.new('RGB',(17,19),(20*i,30,60)).save(p)
                rows.append({'image':str(p),'text':str(i)})
            prototypes=np.eye(8,dtype=np.float32)[:6]
            with contextlib.redirect_stdout(io.StringIO()):
                features,titles,logits,checks=c.extract(backend,rows,prototypes,3,root)
            self.assertEqual(features.shape,(5,8));self.assertEqual(logits.shape,(5,6))
            np.testing.assert_allclose(np.linalg.norm(titles,axis=1),1,atol=1e-6)
            self.assertEqual(len(checks),2)
            self.assertTrue(checks[0]['extra_reference_forward'])
            self.assertFalse(checks[1]['extra_reference_forward'])
            self.assertFalse(backend.model.visual.output_tokens)
            with np.load(root/'patch_verification.npz',allow_pickle=False) as z:
                scores,_=c.score(torch.as_tensor(z['global_features']),torch.as_tensor(z['patches']),torch.as_tensor(z['six_prototypes']))
                np.testing.assert_allclose(scores.numpy(),logits[:3],rtol=1e-6,atol=1e-6)

    def test_calip_cannot_change_gate_and_real_always_maps_true(self):
        x=np.eye(6)[[0,1,2,3,4]];binary=binary_prototypes();six=np.eye(6)*.8
        logits=np.tile(np.array([0,0,0,0,20,0.]),(5,1))
        with contextlib.redirect_stdout(io.StringIO()):
            results,arrays,gates=c.predict(x,x,binary,six,six,logits,c.base_specification(),c.specification(),stata)
            _,_,changed=c.predict(x,x,binary,six,six,-logits,c.base_specification(),c.specification(),stata)
        self.assertEqual(len(results),24)
        self.assertEqual(len({m for m,_,_ in results}),12)
        for seed in [0,1,2]:np.testing.assert_array_equal(gates[seed],changed[seed])
        self.assertEqual(gates[0][0],1)
        for method,seed,pred in results:
            if method.startswith('hierarchy_'):
                self.assertEqual(pred[0],0)
                self.assertTrue(np.all(pred[1:]!=0))
        pred={(m,s):p for m,s,p in results}
        self.assertEqual(pred[('hierarchy_stata_calip_image',0)][1],4)
        self.assertEqual(pred[('hierarchy_stata_clip_image',0)][1],1)
        np.testing.assert_allclose(arrays['p_clip_calip_image'],.5*arrays['p_clip_image']+.5*arrays['p_calip_image'])
        for key,p in arrays.items():
            if key.startswith('p_'):np.testing.assert_allclose(p.sum(1),1)

    def test_singleton_gate_fallback_does_not_call_stata(self):
        x=np.eye(6)[[0]]
        with contextlib.redirect_stdout(io.StringIO()):
            _,_,gates=c.predict(x,x,binary_prototypes(),np.eye(6),np.eye(6),np.zeros((1,6)),
                c.base_specification(),c.specification(),lambda *_a,**_k:self.fail('singleton must keep CLIP'))
        self.assertTrue(all(g[0]==1 for g in gates.values()))

    def test_end_to_end_records_binary_and_sixway_metrics(self):
        from PIL import Image
        original=c.predict
        def mocked(*args):return original(*args,stata_solver=stata)
        def extract(backend,rows,six,batch_size,output):
            x=np.eye(6)[[int(r['text']) for r in rows]]
            np.savez_compressed(output/'patch_verification.npz',synthetic=np.array([1]))
            return x,x,100*x,[{'synthetic':True}]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i in range(5):
                file=root/f'{i}.png';Image.new('RGB',(4,4),(i,0,0)).save(file)
                rows.append({'id':str(i),'image':str(file),'text':str(i),'6way':str(i),'2way':str(int(i==0))})
            settings=config();settings.update(device='cpu',output_root=str(root/'output'))
            with patch.object(c.data,'fakeddit',return_value=(rows,{'cohort_id':'synthetic'})),patch.object(c,'predict',side_effect=mocked),contextlib.redirect_stdout(io.StringIO()):
                output=c.run(settings,'dev','unused',True,FakeBackend,extract)
            run=json.loads((output/'run.json').read_text())
            self.assertEqual(run['kind'],'fakeddit_stata_calip')
            self.assertEqual(len(run['summary']),12)
            for file in ['completed.json','binary_summary.tsv','binary_metrics_by_seed.tsv','comparisons.tsv','oracle_diagnostic.json','extraction.json']:
                self.assertTrue((output/file).exists(),file)
            self.assertFalse(any('oracle' in row['method'] for row in run['summary']))
            self.assertEqual(run['calip_specification']['calip_stage'],2)
        with self.assertRaises(ValueError):c.run({},'test','unused')

    def test_launcher_packages_calip_and_distinct_kernel(self):
        import yaml
        launcher=yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            real=kaggle.path
            def local(v):return Path(tmp) if str(v)=='results/_system/launches/final' else real(v)
            with patch.object(kaggle,'path',side_effect=local),patch.object(kaggle,'cli',side_effect=AssertionError('No API')):
                plan=kaggle.prepare(launcher,'fakeddit_stata_calip_smoke')
            self.assertEqual(plan['task'],'fakeddit_stata_calip')
            self.assertIn('stata-calip-smoke',plan['kernel'])
            self.assertEqual(plan['launcher']['poll_seconds'],30)
            folder=Path(plan['root'])/'kernel'
            manifest=json.loads((folder/'source_manifest.json').read_text())
            self.assertIn('src/vlm_project/calip.py',manifest['files'])
            nb=json.loads((folder/'run.ipynb').read_text())
            source='\n'.join(''.join(cell['source']) for cell in nb['cells'] if cell['cell_type']=='code')
            compile(source,'calip-notebook','exec')
            self.assertIn('calip=True',source)


if __name__=='__main__':unittest.main()
