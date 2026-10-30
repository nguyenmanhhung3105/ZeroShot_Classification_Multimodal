from pathlib import Path
import csv
import tempfile
import unittest
import numpy as np
from experiments.clip_two_stage.multimodal import predict,read_texts,distributions


class MultimodalTests(unittest.TestCase):
    def test_multimodal_launcher_preserves_flag_and_separate_output(self):
        import sys,yaml
        root=Path(__file__).resolve().parents[2]
        sys.path[:0]=[str(root/'scripts'),str(root/'src')]
        import launch_runs as launcher
        config=yaml.safe_load((root/'experiments/clip_two_stage/launcher_dev_multimodal.yaml').read_text())
        launcher.validate(config,['two_stage_mm_dev'])
        launcher.validate_single_cloud_trial(config,['two_stage_mm_dev'])
        with tempfile.TemporaryDirectory() as temp:
            _,_,output,command,_,_,research=launcher.job_plan(config,'two_stage_mm_dev',Path(temp))
            self.assertEqual(output.parts[1],'clip_two_stage_mm_heldout')
            settings=yaml.safe_load(research.read_text())
            self.assertTrue(settings['multimodal'])
            self.assertEqual(settings['max_samples'],2219)
            self.assertEqual(settings['evaluation_scope'],'heldout')

    def test_text_relevance_gate_can_prevent_or_allow_fine_class_change(self):
        info=[[.9,.1],[.9,.1],[.1,.9]]
        human=[[-.1,.003,.002]]*3
        text_info=[[.1,.9],[.9,.1],[.9,.1]]
        text_human=[[-.1,0,.1]]*3
        choices,d=predict(info,human,text_info,text_human,[False]*3,['informative','not_informative'],['not_humanitarian','a','b'],100.)
        self.assertEqual(choices['two_stage_fixed'],['b','b','not_humanitarian'])
        self.assertEqual(choices['two_stage_conditional'],['a','b','not_humanitarian'])
        np.testing.assert_allclose(d['text_weight'],[0,.25,0])

    def test_missing_text_falls_back_to_image_and_not_fake_text_prediction(self):
        choices,d=predict([[.9,.1]],[[-.1,.3,.2]],[[0.,0.]],[[0.,0.,0.]],[True],
                          ['informative','not_informative'],['not_humanitarian','a','b'],100.)
        self.assertEqual(choices['two_stage_fixed'],['a'])
        self.assertEqual(choices['two_stage_conditional'],['a'])
        self.assertEqual(choices['text_only'],['__missing_text__'])
        self.assertEqual(d['text_weight'].tolist(),[0.])

    def test_text_reader_is_bounded_and_checks_id_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rows.tsv'
            with path.open('w',newline='') as f:
                writer=csv.writer(f,delimiter='\t');writer.writerows([['image_id','clean_text'],['1_0','caption'],['2_0','second'],['3_0','ignored']])
            config={'data_path':str(path),'id_col':'image_id','text_col':'clean_text'}
            texts,identity=read_texts(config,[{'id':'1_0'},{'id':'2_0'}],Path)
            self.assertEqual(texts,['caption','second'])
            self.assertEqual(len(identity),2)
            with self.assertRaises(ValueError):read_texts(config,[{'id':'2_0'}],Path)

    def test_invalid_distributions_rejected(self):
        with self.assertRaises(ValueError):distributions([[float('nan'),0]],100.)
        with self.assertRaises(ValueError):distributions([[1.,0]],-1.)


if __name__=='__main__':unittest.main()
