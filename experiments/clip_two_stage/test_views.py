from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from experiments.clip_two_stage.views import full_frame,classify_views


class ViewTests(unittest.TestCase):
    def test_full_frame_preserves_edges_pixels_and_aspect(self):
        image=Image.new('RGB',(8,4),'red');image.putpixel((0,0),(0,255,0))
        square=full_frame(image)
        self.assertEqual(square.size,(8,8))
        np.testing.assert_array_equal(np.asarray(square)[2:6],np.asarray(image))
        self.assertEqual(square.getpixel((0,0)),(0,0,0))
        square.close();image.close()

    def test_identical_views_preserve_original_decisions(self):
        from experiments.clip_two_stage.run import decide
        scores=[np.array([[.1,.9],[.8,.2]]),np.array([[.9,.1,.2],[.1,.2,.3]])]
        il=['informative','not_informative'];labels=['not_humanitarian','a','b']
        result=classify_views(scores,scores,il,labels)
        self.assertEqual(result['mean_view_two_stage'],decide(*scores,il,labels)[0])
        self.assertEqual(result['full_frame_two_stage'],result['mean_view_two_stage'])

    def test_views_launcher_freezes_flag_and_output(self):
        import yaml
        root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'scripts'),str(root/'src')]
        import launch_runs as launcher
        config=yaml.safe_load((root/'experiments/clip_two_stage/launcher_dev_views.yaml').read_text())
        launcher.validate(config,['two_stage_views_dev'])
        with tempfile.TemporaryDirectory() as temp:
            _,_,output,_,_,_,research=launcher.job_plan(config,'two_stage_views_dev',Path(temp))
            self.assertEqual(output.parts[1],'clip_two_stage_views_heldout')
            self.assertTrue(yaml.safe_load(research.read_text())['full_frame_view'])


if __name__=='__main__':unittest.main()
