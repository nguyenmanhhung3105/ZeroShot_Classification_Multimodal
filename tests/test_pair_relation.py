"""Kiểm quan hệ theo nhãn và toàn luồng bằng dữ liệu giả, không tải checkpoint."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference import predict_single_label, predict_multi_label
from pair_relation import (validate_relation_options, relation_settings, relation_evidence,
                           apply_class_relation, attach_relation)
from text_chunking import encode_chunked_texts
from test_runtime_fixes import FakeVLM
from test_text_chunking import ChunkVLM
from logging_utils import save_single_label_log, save_multi_label_log
from research.run_relation_ablation import bounded_copy, compare_rows


class PairRelationTests(unittest.TestCase):
    def test_scope_and_policies_follow_taxonomy(self):
        opts = {"enabled": True, "mode": "class_aware"}
        self.assertEqual(relation_settings({"task": "6way", "image_text_relation": opts}, "fakeddit")[0]["class_policies"],
                         {0: "match", 2: "mismatch"})
        for dataset, task in (("fakeddit", "2way"), ("crisismmd", "humanitarian"), ("mmimdb", "genres")):
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                relation_settings({"task": task, "image_text_relation": opts}, dataset)
        for policies in ({4: "mismatch"}, {1: "match"}, {True: "match"}, {0: "match", "0": "match"}):
            with self.assertRaises(ValueError):
                validate_relation_options({**opts, "class_policies": policies})

    def test_invalid_configuration(self):
        for config in (False, {"enabled": "true"}, {"mode": "mismatch"}, {"unknown": True},
                       {"weight": -1}, {"weight": 2}, {"weight": float("nan")}, {"weight": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_relation_options(config)

    def test_evidence_extremes_and_missing_vectors(self):
        image = np.array([[1., 0.]] * 4)
        text = np.array([[1., 0.], [-1., 0.], [0., 0.], [1., 0.]])
        pi = np.array([[1., 0.]] * 4)
        pt = np.array([[1., 0.], [0., 1.], [.5, .5], [1., 0.]])
        scores = relation_evidence(image, text, pi, pt, [False, False, False, True])
        self.assertEqual(scores["image_text_similarity"], [1., -1., None, None])
        self.assertEqual(scores["image_text_js_divergence"][:2], [0., 1.])
        self.assertEqual(scores["image_text_match_evidence"][:2], [1., 0.])
        self.assertEqual(scores["image_text_mismatch_evidence"][:2], [0., 1.])

    def test_class_formula_can_change_match_and_mismatch_candidates(self):
        labels = [2, 5, 0, 3, 1, 4]  # kiểm thứ tự khác ID mặc định
        base = np.array([[.30, .10, .29, .10, .10, .11], [.29, .10, .30, .10, .10, .11]])
        scores = {"image_text_match_evidence": [1., 0.], "image_text_mismatch_evidence": [0., 1.]}
        options = validate_relation_options({"enabled": True, "mode": "class_aware", "weight": .25})
        result = apply_class_relation(base, labels, scores, options, .75)
        self.assertEqual([labels[x] for x in result.argmax(axis=1)], [0, 2])
        expected = base.copy()
        expected[0, 2] *= 1.25
        expected[1, 0] *= 1.25
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result.sum(axis=1), 1)
        np.testing.assert_allclose(result[:,1] / result[:,3], base[:,1] / base[:,3])
        self.assertEqual(scores["relation_applied"], [True, True])

    def test_uniform_agreement_has_no_relation_evidence(self):
        uniform=np.full((1,6),1/6)
        scores=relation_evidence([[1,0]],[[1,0]],uniform,uniform,[False])
        self.assertAlmostEqual(scores["image_text_reliability"][0],0)
        self.assertAlmostEqual(scores["image_text_match_evidence"][0],0)
        self.assertAlmostEqual(scores["image_text_mismatch_evidence"][0],0)
        opts=validate_relation_options({"enabled":True,"mode":"class_aware"})
        np.testing.assert_array_equal(apply_class_relation(uniform,list(range(6)),scores,opts,.5),uniform)
        self.assertEqual(scores["relation_applied"],[False])

    def test_zero_weight_missing_and_single_modality_preserve_exact_scores(self):
        base = np.array([[.2, .1, .3, .1, .1, .2]])
        opts = validate_relation_options({"enabled": True, "mode": "class_aware"})
        for options, weight, evidence in (({**opts, "weight": 0}, .5, 1), (opts, 0, 1),
                                          (opts, 1, 1), (opts, .5, None)):
            scores = {"image_text_match_evidence": [evidence], "image_text_mismatch_evidence": [evidence]}
            np.testing.assert_array_equal(apply_class_relation(base, list(range(6)), scores, options, weight), base)

    def test_invalid_probabilities_and_embeddings_are_rejected(self):
        with self.assertRaises(ValueError):
            relation_evidence([[float("nan"), 0]], [[1, 0]], [[1,0]], [[1,0]], [False])
        with self.assertRaises(ValueError):
            relation_evidence([[1,0]], [[1,0]], [[2,0]], [[1,0]], [False])
        opts = validate_relation_options({"enabled": True, "mode": "class_aware"})
        with self.assertRaises(ValueError):
            apply_class_relation(np.ones((1,6)), list(range(6)), {}, opts, .5)

    def test_diagnostic_preserves_all_outputs_for_each_weight(self):
        for weight in (0, .5, 1):
            args = ([None] * 2, ["text", ""], np.eye(2), [0,1])
            baseline_model, relation_model = FakeVLM(), FakeVLM()
            baseline = predict_single_label(baseline_model, *args, image_weight=weight, show_progress=False)
            buffer = {}
            result = predict_single_label(relation_model, *args, image_weight=weight, show_progress=False,
                image_text_relation={"enabled": True}, relation_scores=buffer)
            self.assertEqual(result[0], baseline[0])
            for a,b in zip(result[1:],baseline[1:]):
                np.testing.assert_array_equal(a,b)
            self.assertAlmostEqual(buffer["image_text_similarity"][0], 0)
            self.assertIsNone(buffer["image_text_similarity"][1])
            self.assertEqual(sum(relation_model.image_batches), 2)
            self.assertEqual(sum(relation_model.text_batches), 1)

    def test_class_aware_inference_is_consistent_with_formula(self):
        labels = list(range(6))
        classes = np.array([[1.,0.],[0.,1.],[.7,.7],[-1.,0.],[0.,-1.],[-.7,-.7]])
        opts = {"enabled": True, "mode": "class_aware"}
        args = ([None,None], ["text",""], classes, labels)
        baseline = predict_single_label(FakeVLM(), *args, show_progress=False)
        buffer, audit = {}, {}
        result = predict_single_label(FakeVLM(), *args, show_progress=False, image_text_relation=opts,
                                      relation_scores=buffer, diagnostics=audit)
        expected = apply_class_relation(baseline[1],labels,deepcopy(buffer),validate_relation_options(opts),.5)
        np.testing.assert_allclose(result[1],expected)
        self.assertEqual(result[0], [labels[i] for i in result[1].argmax(axis=1)])
        np.testing.assert_array_equal(result[1][1],baseline[1][1])
        np.testing.assert_array_equal(result[2],baseline[2])
        self.assertEqual(audit["image_text_relation"]["class_policies"], {0:"match",2:"mismatch"})

    def test_multilabel_chunk_modes_reuse_encoding_and_preserve_predictions(self):
        texts = ['Long plot. Another part. The final scene.', '']
        for mode in ("relative_z","fixed_threshold"):
            for aggregation in ("mean_embed","max_sim"):
                for weight in (0,.5,1):
                    kwargs = dict(decision_mode=mode, chunk_long_text=True, chunk_aggregation=aggregation,
                                  image_weight=weight, show_progress=False)
                    baseline_model, model = ChunkVLM(), ChunkVLM()
                    args = ([None,None], texts, np.eye(2), ["a","b"])
                    baseline = predict_multi_label(baseline_model,*args,**kwargs)
                    buffer = {}
                    result = predict_multi_label(model,*args,**kwargs, image_text_relation={"enabled":True},
                                                 relation_scores=buffer)
                    self.assertEqual(result[0],baseline[0])
                    for a,b in zip(result[1:],baseline[1:]):
                        np.testing.assert_array_equal(a,b)
                    vector,_ = encode_chunked_texts(ChunkVLM(),texts[:1],16)
                    self.assertAlmostEqual(buffer["image_text_similarity"][0],vector[0,0],places=9)
                    self.assertIsNone(buffer["image_text_similarity"][1])
                    if weight<1:
                        self.assertEqual(model.text_batches,baseline_model.text_batches)

    def test_log_relation_with_audit_disabled_for_both_loggers(self):
        frame=pd.DataFrame({"id":["one","two"],"text":["text",""],"image_path":["x","y"]})
        attach_relation(frame,{"image_text_similarity":[.3,None],"relation_applied":[True,False]})
        for multi in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                save=save_multi_label_log if multi else save_single_label_log
                truth=[["a"],["b"]] if multi else ["a","b"]
                output=Path(save(frame,truth,truth,np.array([[.8,.2],[.2,.8]]),["a","b"],
                    "id","text","stub","fake","task",directory,
                    extra_manifest={"relation_enabled":True}))
                logged=pd.read_csv(output/"raw_predictions.tsv",sep="\t",nrows=2)
                self.assertAlmostEqual(logged.image_text_similarity.iloc[0],.3)
                self.assertTrue(pd.isna(logged.image_text_similarity.iloc[1]))
                self.assertEqual(logged.relation_applied.tolist(),[True,False])
                self.assertTrue(json.loads((output/"manifest.json").read_text())["relation_enabled"])

    def test_runner_integrates_relation_metadata_and_correction(self):
        from run import run_fakeddit as module
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"synthetic.tsv"
            pd.DataFrame({"id":["one","two"],"clean_title":["long title for a synthetic sample", "another synthetic title"],
                          "6_way_label":[0,2]}).to_csv(path,sep="\t",index=False)
            task={"task":"6way","data_path":str(path),"max_samples":2,
                  "image_text_relation":{"enabled":True,"mode":"class_aware"}}
            cfg={"output":{"raw_predictions_dir":directory},"evaluation":{"enabled":False}}
            with patch.object(module,"resolve_image_paths",return_value=["unused"]*2), \
                 patch.object(module,"load_images_safe",return_value=([None,None],[0,1])):
                result=module.run_fakeddit(FakeVLM(),task,cfg)
            manifest=json.loads((Path(result["prediction_log_dir"])/"manifest.json").read_text())
            self.assertEqual(manifest["relation_mode"],"class_aware")
            rows=pd.read_csv(Path(result["prediction_log_dir"])/"raw_predictions.tsv",sep="\t",nrows=2)
            self.assertTrue(np.isfinite(rows.image_text_similarity).all())
            self.assertEqual(manifest["relation_algorithm"],"cosine_js_confidence_bounded_v1")

    def test_smoke_copies_only_first_five_before_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            source,target=Path(directory)/"source.tsv",Path(directory)/"bounded.tsv"
            source.write_text('id\tclean_title\n'+'\n'.join(f'{i}\ttext' for i in range(20)))
            self.assertEqual(bounded_copy(source,target),5)
            with target.open() as stream:
                rows=list(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual([r['id'] for r in rows],list(map(str,range(5))))

    def test_smoke_comparison_uses_same_logged_probabilities(self):
        rows=[]
        for i in range(2):
            p=[.2,.1,.3,.1,.1,.2]
            rows.append({'id':str(i),'y_true':'2','image_text_similarity':'0.2',
                **{f'score_{c}':str(x) for c,x in enumerate(p)},
                **{f'score_image_{c}':str(x) for c,x in enumerate(p)},
                **{f'score_text_{c}':str(x) for c,x in enumerate(p)}})
        opts=validate_relation_options({'enabled':True,'mode':'class_aware'})
        summary,comparison=compare_rows(rows,opts,.75)
        self.assertEqual(len(summary),3)
        self.assertEqual(len(comparison),6)
        self.assertTrue(all(r['n_samples']==2 for r in summary))
        self.assertEqual(summary[0]['accuracy'],1.)


if __name__ == '__main__':
    unittest.main()
