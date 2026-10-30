import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.crisis_reasoning.hierarchy import reconcile
from experiments.crisis_reasoning import run as runner


class ReasoningTests(unittest.TestCase):
    def test_routing_uses_disagreement_and_invalid_review_falls_back(self):
        from experiments.crisis_reasoning.cascade import route
        output,flags=route(['a','a','b'],['a','b','a'],['b','b','__invalid__'],['a','b'])
        self.assertEqual(flags,[False,True,True])
        self.assertEqual(output,['a','b','b'])

    def test_hierarchy_preserves_group_mass_and_conditional_ratios(self):
        out = reconcile([[.7,.3]], [[.4,.2,.4]], ['informative','not_informative'],
                        ['not_humanitarian','a','b'])
        np.testing.assert_allclose(out, [[.3,.7/3,1.4/3]])
        np.testing.assert_allclose(out.sum(1),1)
        fallback = reconcile([[1.,0.]], [[1.,0.,0.]], ['informative','not_informative'],
                             ['not_humanitarian','a','b'])
        np.testing.assert_allclose(fallback,[[0.,.5,.5]])

    def test_rejects_invalid_probabilities_and_labels(self):
        with self.assertRaises(ValueError):
            reconcile([[2.,-1.]], [[.5,.5]], ['informative','not_informative'],['not_humanitarian','a'])
        for response in ['not_informative', '{"label":"made_up","evidence":"x"}',
                         '{"label":"informative"}', '{"label":"informative","evidence":""}']:
            self.assertIsNone(runner.parse_response(response,['informative','not_informative']))
        self.assertEqual(runner.parse_response('{"label":"informative","evidence":"flood"}',
                                               ['informative'])['label'],'informative')

    def test_prompts_contain_all_classes_and_no_sample_fields(self):
        for task in ['informativeness','humanitarian']:
            text=runner.prompt(task)
            for label in runner.get_label_names(task):
                self.assertIn(label+':',text)
            for forbidden in ['y_true','clean_text','image_path','tweet_id']:
                self.assertNotIn(forbidden,text)

    def test_prepare_does_not_read_data_or_load_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner,'project_path',side_effect=Path), \
             patch.object(runner,'read_cohort',side_effect=AssertionError('dataset')):
            runner.run({'task':'informativeness','max_samples':5,'base_config':'unused',
                        'output_dir':tmp},True,backend_factory=lambda: self.fail('model'))
            self.assertTrue((Path(tmp)/'frozen_plan.json').exists())

    def test_invalid_generation_counts_as_error_and_backend_closes(self):
        class Backend:
            revision='synthetic'
            closed=False
            def predict(self,image,text): return 'invalid output'
            def close(self): self.closed=True
        backend=Backend()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(8,8)).save(root/'image.png')
            (root/'base.yaml').write_text('datasets:\n  crisismmd: {}\n')
            rows=[{'id':'one','y_true':'informative','image_path':str(root/'image.png')}]
            with patch.object(runner,'project_path',side_effect=Path), \
                 patch.object(runner,'read_cohort',return_value=rows), \
                 patch.object(runner,'lock_cohort',return_value={'cohort_fingerprint':'fake'}):
                out=runner.run({'task':'informativeness','max_samples':1,'base_config':str(root/'base.yaml'),
                                'output_dir':str(root/'output')},backend_factory=lambda:backend)
            metrics=json.loads((out/'metrics.json').read_text())
            self.assertEqual(metrics['invalid_count'],1)
            self.assertEqual(metrics['accuracy'],0)
            self.assertTrue(backend.closed)
            with self.assertRaises(FileExistsError):
                with patch.object(runner,'project_path',side_effect=Path):
                    runner.run({'task':'informativeness','max_samples':1,'base_config':'unused',
                                'output_dir':str(out)},True)

    def test_launcher_uses_qwen_command_and_output_without_model_load(self):
        import sys
        sys.path.insert(0,str(runner.ROOT/'scripts'))
        import launch_runs as launcher
        import yaml
        config=yaml.safe_load((runner.ROOT/'experiments/crisis_reasoning/launcher.yaml').read_text())
        launcher.validate(config,['qwen_info'])
        launcher.validate_single_cloud_trial(config,['qwen_info'])
        with tempfile.TemporaryDirectory() as tmp:
            _,_,output,command,_,_,_=launcher.job_plan(config,'qwen_info',Path(tmp))
            self.assertEqual(command[0],'experiments/crisis_reasoning/run.py')
            self.assertEqual(output.parts[-2],'qwen2_5_vl_3b')
            self.assertIn('qwen_direct_image',output.parts)

    def test_archive_contains_reasoning_dependencies(self):
        import sys
        import zipfile
        sys.path.insert(0,str(runner.ROOT/'scripts'))
        import launch_runs as launcher
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'code.zip'
            launcher.archive_code(archive)
            with zipfile.ZipFile(archive) as stream:
                self.assertIn('requirements/vlm_reasoning.txt',stream.namelist())
                self.assertIn('experiments/crisis_reasoning/run.py',stream.namelist())


if __name__=='__main__':
    unittest.main()
