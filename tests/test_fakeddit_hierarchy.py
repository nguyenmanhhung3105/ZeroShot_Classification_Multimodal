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
from vlm_project import fakeddit_hierarchy as h, kaggle
from vlm_project.io import config


def binary_prototypes():
    result=np.zeros((2,6));result[0,1:]=1/np.sqrt(5);result[1,0]=1
    return result*.8


def em(q,**kwargs):return q.argmax(-1),{},q


def stata(x,t,**kwargs):
    p=softmax(100*x@t.T,axis=1)
    return p,p


def legacy(x,raw,seed):
    order=np.random.default_rng(seed).permutation(len(x));groups=np.empty(len(x),dtype=int)
    for group,start in enumerate(range(0,len(x),75)):groups[order[start:start+75]]=group
    pred=(x@h.unit(raw).T).argmax(1)
    return pred,{'batch_ids':groups,'votes':np.stack([pred,pred,pred],axis=1)}


class FakeBackend:
    metadata={'backend':'synthetic-only'}
    def __init__(self,device):pass
    def prototypes(self,bank):return binary_prototypes() if len(bank)==2 else np.eye(6)*.8
    def encode_images(self,images):return np.eye(6)[[im.getpixel((0,0))[0] for im in images]]
    def encode_texts(self,texts,strict=False):return np.eye(6)[[int(t) for t in texts]]
    def unload(self):pass


class FakedditHierarchyTests(unittest.TestCase):
    def test_original_reference_matches_existing_fakeddit_prediction(self):
        from vlm_project.fakeddit import predict as original_predict
        # Singleton exercises the original binary fallback without fitting real data.
        x=np.eye(6)[[3]];binary=binary_prototypes();six=np.eye(6)*.8
        profile=json.loads((ROOT/'configs/fakeddit_prompts.json').read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            results,_,_=h.predict(x,x,binary,six,six,h.specification(),em,stata)
            old,_=original_predict(x,x,{'2way':{'legacy':binary,'semantic_v1':binary},'6way':{'legacy':six}},
                                   {'2way':binary,'6way':six},profile)
        expected={seed:pred.astype(int) for task,method,seed,pred in old if task=='6way' and method=='proposed'}
        for method,seed,pred in results:
            if method=='original_hierarchy_mm':np.testing.assert_array_equal(pred,expected[seed])

    def test_shared_partition_and_original_mm_controls(self):
        x=np.eye(6)[np.arange(80)%6]
        with contextlib.redirect_stdout(io.StringIO()):
            results,arrays,gates=h.predict(x,x,binary_prototypes(),np.eye(6)*.8,np.eye(6)*.8,h.specification(),em,stata,legacy)
        self.assertEqual(len(results),25)
        self.assertEqual(len({name for name,_,_ in results}),11)
        self.assertEqual(len(gates),12)
        for seed in [0,1,2]:
            self.assertEqual(sorted(np.bincount(arrays[f'groups_s{seed}']).tolist()),[5,75])
        np.testing.assert_array_equal(h.multiclass_vote(np.array([[0,1,2],[0,1,1]]),6),[0,1])

    def test_dev_only_and_end_to_end_oracle_excluded(self):
        with self.assertRaises(ValueError):h.run({},'test','unused')
        original=h.predict
        def mocked(*args):return original(*args,em_solver=em,stata_solver=stata,legacy_solver=legacy)
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rows=[]
            for i,label in enumerate([0,1,2,3,5]):
                image=folder/f'{i}.png';Image.new('RGB',(4,4),(label,0,0)).save(image)
                rows.append({'id':str(i),'image':str(image),'text':str(label),'6way':str(label),'2way':str(int(label==0))})
            settings=config();settings.update(device='cpu',output_root=str(folder/'output'))
            with patch.object(h.data,'fakeddit',return_value=(rows,{'cohort_id':'synthetic'})), \
                    patch.object(h,'predict',side_effect=mocked),contextlib.redirect_stdout(io.StringIO()):
                output=h.run(settings,'dev','unused',True,FakeBackend)
            result=json.loads((output/'run.json').read_text())
            self.assertEqual(len(result['summary']),11)
            self.assertEqual(result['samples'],5)
            self.assertTrue((output/'completed.json').exists())
            self.assertTrue((output/'oracle_diagnostic.json').exists())
            self.assertFalse(any('oracle' in row['method'] for row in result['summary']))
            with np.load(output/'replay_scores.npz',allow_pickle=False) as z:
                self.assertEqual(len(z['labels']),5)

    def test_launcher_whitelist_and_no_network_prepare(self):
        import yaml
        launcher=yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            real=kaggle.path
            def local(v):return Path(tmp) if str(v)=='results/_system/launches/final' else real(v)
            with patch.object(kaggle,'path',side_effect=local),patch.object(kaggle,'cli',side_effect=AssertionError('No API')):
                plan=kaggle.prepare(launcher,'fakeddit_hierarchy_smoke')
            self.assertEqual(plan['samples'],5)
            self.assertEqual(plan['launcher']['poll_seconds'],30)
            self.assertEqual(plan['data_dataset'],'nguyenhung3105/vlm-fakeddit-dev-v2')
            kernel=Path(plan['root'])/'kernel'
            manifest=json.loads((kernel/'source_manifest.json').read_text())
            self.assertIn('run.py',manifest['files'])
            self.assertNotIn('src/vlm_project/paper.py',manifest['files'])
            for cell in json.loads((kernel/'run.ipynb').read_text())['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),'fakeddit-notebook','exec')


if __name__=='__main__':unittest.main()
