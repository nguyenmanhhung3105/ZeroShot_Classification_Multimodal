"""Train-CV và phân tích lỗi chỉ dùng dữ liệu/cache tổng hợp."""
import contextlib
import io as stdio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_project import fewshot_task1_analysis as analysis
from vlm_project import fakeddit_fewshot_task1 as fs
from vlm_project import io,evaluate,protocol


class FewshotAnalysisTests(unittest.TestCase):
    def test_diagnostics_are_binary_detection_not_subtype_classification(self):
        six=np.array([0,0,2,2,3,4])
        good=np.tile([0,0,1,1,1,1],(5,1))
        image=good.copy();image[:,2]=0
        joint=good.copy();joint[:,0]=1
        predictions={'image':image,'title':good,'image_title':joint}
        sub,persistent,changes=analysis.diagnostics(six,predictions,np.array(list('abcdef')))
        self.assertEqual(next(r for r in sub if r['method']=='image' and r['subtype']=='False Connection')['binary_detection_recall_mean'],.5)
        self.assertEqual(next(r for r in sub if r['method']=='image_title' and r['subtype']=='False Connection')['binary_detection_recall_mean'],1.)
        self.assertEqual(persistent[0]['id'],'a')
        self.assertEqual(persistent[0]['wrong_seeds'],5)
        self.assertEqual(next(r for r in changes if r['class']=='Fake')['image_wrong_to_correct'],1)

    def test_end_to_end_train_only_selection_and_unchanged_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            spec=fs.specification()
            y6=np.repeat(np.arange(6),64);dev6=np.tile(np.arange(6),500)
            image=np.eye(8)[y6];title=np.roll(image,1,axis=1)
            dev_image=np.eye(8)[dev6];dev_title=np.roll(dev_image,1,axis=1)
            train_ids=np.asarray([f't{i}' for i in range(384)])
            dev_ids=np.asarray([f'd{i}' for i in range(3000)])
            supports={seed:fs.select_support(y6,64,seed) for seed in range(5)}
            log=[{'seed':s,'ids':train_ids[ix].tolist(),'labels_6way':y6[ix].tolist()} for s,ix in supports.items()]
            io.write_json(source/'support.json',log)
            io.write_json(source/'cohort.json',[{'id':str(i),'6way':str(y)} for i,y in zip(dev_ids,dev6)])
            np.savez_compressed(source/'features.npz',train_image=image,train_title=title,
                                dev_image=dev_image,dev_title=dev_title,train_y6=y6,
                                train_ids=train_ids,dev_ids=dev_ids,**{f'support_s{s}':ix for s,ix in supports.items()})
            with contextlib.redirect_stdout(stdio.StringIO()):
                results,_=fs.predict(image,title,y6,dev_image,dev_title,supports,spec)
            truth=(dev6!=0).astype(int);rows=[];probabilities={}
            for method,seed,p in results:
                m=evaluate.metric(truth,p.argmax(1),[0,1]);m.pop('per_class')
                rows.append({'task':'2way','method':method,'seed':seed,'samples':3000,**m})
                probabilities[f'p_{method}_s{seed}']=p
            np.savez_compressed(source/'replay_scores.npz',ids=dev_ids,labels=truth,**probabilities)
            with contextlib.redirect_stdout(stdio.StringIO()):
                evaluate.publish(source,rows,[{'id':'synthetic'}],{},
                                 {'kind':'fakeddit_fewshot_task1','mode':'full','samples':3000,
                                  'model':'synthetic','test_opened':False,'specification':spec,
                                  'support_sha256':io.digest(log)})
            before=io.sha256(source/'run.json')
            with patch.object(analysis,'probe',wraps=analysis.probe) as probe,contextlib.redirect_stdout(stdio.StringIO()):
                output=analysis.analyze(source,root/'review')
            self.assertEqual(probe.call_count,80)  # 5 seed × (5 C × 3 fold + 1 refit).
            self.assertEqual({len(c.args[0]) for c in probe.call_args_list},{256,384})
            self.assertEqual(io.sha256(source/'run.json'),before)
            protocol.completed_run(source)
            reviewed=protocol.completed_run(output)
            self.assertEqual(len(reviewed['summary']),4)
            self.assertTrue(all(r['seeds']==5 for r in reviewed['summary']))
            import csv
            with (output/'selected_C.tsv').open() as f:
                selected=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual([float(r['selected_C']) for r in selected],[1.]*5)
            with np.load(output/'replay_scores.npz',allow_pickle=False) as a:
                for seed in range(5):
                    np.testing.assert_allclose(a[f'p_image_title_train_cv_s{seed}'],a[f'p_image_title_s{seed}'])


if __name__=='__main__':unittest.main()
