"""Flag quan hệ trên mọi task/runner; chỉ dùng dữ liệu và embedding tổng hợp."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_relation import relation_settings
from inference import predict_single_label
from research import run_upstream as upstream
from research.run_adapted_methods import prepare_trials
from test_runtime_fixes import FakeVLM
import test_upstream_projects as fixtures
SyntheticBackend = fixtures.SyntheticBackend

CASES = (("fakeddit", "2way"), ("fakeddit", "6way"), ("crisismmd", "informativeness"),
         ("crisismmd", "humanitarian"), ("mmimdb", "genres"))


class RelationFlagTests(unittest.TestCase):
    def test_default_disabled_and_global_auto_for_all_tasks(self):
        for dataset, task in CASES:
            active = {"task": task, "image_text_relation": {"mode": "auto"}}
            options, _ = relation_settings(active, dataset)
            self.assertFalse(options["enabled"])
            config = {"inference": {"image_text_relation": {"enabled": True, "mode": "auto"}}}
            options, metadata = relation_settings(active, dataset, config)
            self.assertTrue(options["enabled"])
            expected = "class_aware" if (dataset, task) == ("fakeddit", "6way") else "diagnostic"
            self.assertEqual(options["mode"], expected)
            self.assertEqual(metadata["relation_mode_requested"], "auto")
            self.assertEqual(options["class_policies"], {0:"match",2:"mismatch"} if expected == "class_aware" else {})

    def test_dataset_override_and_invalid_flag(self):
        config = {"inference":{"image_text_relation":{"enabled":True,"mode":"auto","weight":.25}}}
        task = {"task":"6way","image_text_relation":{"enabled":False,"weight":.1}}
        options, _ = relation_settings(task,"fakeddit",config)
        self.assertFalse(options["enabled"])
        self.assertEqual(options["weight"],.1)
        for layer in (False, [], "true"):
            with self.assertRaises(ValueError):
                relation_settings({"task":"6way","image_text_relation":layer},"fakeddit")
            with self.assertRaises(ValueError):
                relation_settings({"task":"6way"},"fakeddit",{"inference":{"image_text_relation":layer}})

    def test_shipped_flags_are_all_off_without_dataset_enabling(self):
        root=Path(__file__).resolve().parents[1]
        config=yaml.safe_load((root/"configs/experiment_config.yaml").read_text())
        for dataset, task in config["datasets"].items():
            self.assertFalse(relation_settings(task,dataset,config)[0]["enabled"])
        for name in ("adapted","upstream"):
            settings=yaml.safe_load((root/f"configs/research/{name}.yaml").read_text())
            self.assertIs(settings["image_text_relation"]["enabled"],False)

    def test_disabled_flag_preserves_encoders_and_scores_on_backbones(self):
        labels=list(range(6))
        classes=np.array([[1.,0.],[0.,1.],[.7,.7],[-1.,0.],[0.,-1.],[-.7,-.7]])
        for name in ("metaclip2_vith14","dfn5b_vith14","siglip2_so400m"):
            for weight in (0,.5,1):
                baseline, flagged=FakeVLM(),FakeVLM()
                baseline.model_name=flagged.model_name=name
                args=([None,None],["text",""],classes,labels)
                old=predict_single_label(baseline,*args,image_weight=weight,show_progress=False)
                options,_=relation_settings({"task":"6way"},"fakeddit",
                    {"inference":{"image_text_relation":{"enabled":False,"mode":"auto"}}})
                scores={}
                new=predict_single_label(flagged,*args,image_weight=weight,show_progress=False,
                                         image_text_relation=options,relation_scores=scores)
                self.assertEqual(old[0],new[0])
                for a,b in zip(old[1:],new[1:]):
                    np.testing.assert_array_equal(a,b)
                self.assertEqual(baseline.image_batches,flagged.image_batches)
                self.assertEqual(baseline.text_batches,flagged.text_batches)
                self.assertEqual(scores,{})

    def test_adapted_override_resolves_all_tasks_with_one_flag(self):
        for dataset,task in CASES:
            base={"datasets":{dataset:{"task":task}},
                  "inference":{"image_text_relation":{"enabled":True,"mode":"auto"}}}
            settings={"dataset":dataset,"task":task,"methods":["baseline"],"max_samples":2,
                      "image_text_relation":{"enabled":False,"mode":"auto"}}
            trial=prepare_trials(settings,base)[0]
            self.assertFalse(relation_settings(trial,dataset,base)[0]["enabled"])
            settings["image_text_relation"]["enabled"]=True
            trial=prepare_trials(settings,base)[0]
            self.assertTrue(relation_settings(trial,dataset,base)[0]["enabled"])


class UpstreamRelationFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.UpstreamProjectTests.setUpClass()
        cls.fixture_owner=fixtures.UpstreamProjectTests()
        cls.api=fixtures.UpstreamProjectTests.api
        cls.sources=fixtures.UpstreamProjectTests.sources

    def test_enabled_relation_logs_all_five_tasks_and_preserves_native(self):
        for dataset,task in CASES:
            with self.subTest(dataset=dataset,task=task), tempfile.TemporaryDirectory() as directory:
                config,base,settings=self.fixture_owner.fixture(Path(directory),dataset,task)
                native_baseline=None
                for enabled in (False,True):
                    active=deepcopy(settings)
                    active["output_dir"]=str(Path(directory)/f'results_{enabled}')
                    active["image_text_relation"]={"enabled":enabled,"mode":"auto","weight":.25}
                    config.write_text(json.dumps(active))
                    with patch.object(upstream,"OpenAIClipBackend",SyntheticBackend), \
                         redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                        result=upstream.main(["--config",str(config)])
                    summary=pd.read_csv(result/"summary.tsv",sep='\t')
                    self.assertTrue(summary.relation_enabled.eq(enabled).all())
                    native=summary[summary.output_variant=='original_image']
                    metrics=[c for c in ('accuracy','macro_f1','micro_f1','samples_f1') if c in native]
                    if not enabled:
                        native_baseline=native.set_index(['method','seed'])[metrics]
                    else:
                        pd.testing.assert_frame_equal(native.set_index(['method','seed'])[metrics],native_baseline)
                        self.assertTrue(native.relation_mode.eq('diagnostic').all())
                    for row in summary.itertuples():
                        log=Path(row.prediction_log_dir)
                        predictions=pd.read_csv(log/'raw_predictions.tsv',sep='\t',nrows=3)
                        manifest=json.loads((log/'manifest.json').read_text())
                        self.assertEqual(manifest['relation_enabled'],enabled)
                        self.assertEqual('image_text_similarity' in predictions,enabled)
                        if enabled:
                            self.assertTrue(np.isfinite(predictions.image_text_similarity.iloc[0]))
                            self.assertTrue(pd.isna(predictions.image_text_similarity.iloc[1]))
                            if row.output_variant=='original_image':
                                self.assertFalse(predictions.relation_applied.any())
                            expected='class_aware' if dataset=='fakeddit' and task=='6way' and row.output_variant=='multimodal' else 'diagnostic'
                            self.assertEqual(row.relation_mode,expected)
                    self.assertEqual(sum(SyntheticBackend.instances[-1].image_batches),2)

    def test_native_only_off_skips_text_and_on_encodes_relation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,base,settings=self.fixture_owner.fixture(root,'fakeddit','6way')
            settings['outputs']=['original_image']
            for enabled in (False,True):
                settings['image_text_relation']={'enabled':enabled,'mode':'auto'}
                specs,_=upstream.prepare_plan(settings,base,self.api,root,self.sources)
                spec=specs['fakeddit']
                cohort=upstream.load_cohort('fakeddit',spec)
                backend=SyntheticBackend(self.api,'ViT-B/32','cpu')
                cache=upstream.encode_inputs(backend,'fakeddit',spec,cohort,base,settings)
                self.assertEqual('relation_embeddings' in cache,enabled)
                if not enabled:
                    self.assertEqual(backend.text_batches,[])

    def test_max_sim_relation_reuses_chunk_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,base,settings=self.fixture_owner.fixture(root,'mmimdb','genres')
            base['datasets']['mmimdb']['chunk_aggregation']='max_sim'
            caches=[]
            backends=[]
            for enabled in (False,True):
                settings['image_text_relation']={'enabled':enabled,'mode':'auto'}
                specs,_=upstream.prepare_plan(settings,base,self.api,root,self.sources)
                spec=specs['mmimdb']
                cohort=upstream.load_cohort('mmimdb',spec)
                backend=SyntheticBackend(self.api,'ViT-B/32','cpu')
                caches.append(upstream.encode_inputs(backend,'mmimdb',spec,cohort,base,settings))
                backends.append(backend)
            np.testing.assert_array_equal(caches[0]['text_scores'],caches[1]['text_scores'])
            self.assertEqual(backends[0].text_batches,backends[1].text_batches)
            vector=caches[1]['relation_embeddings']['text'][0]
            self.assertAlmostEqual(np.linalg.norm(vector),1)


if __name__=='__main__':
    unittest.main()
