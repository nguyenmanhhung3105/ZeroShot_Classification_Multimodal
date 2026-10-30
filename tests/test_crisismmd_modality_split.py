import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prompts import crisismmd_prompts as prompts
from run.run_crisismmd import prepare_prompt_embeddings
from run.common import prompt_trial_configs
from results_layout import job_path


class ModalitySplitTests(unittest.TestCase):
    def test_partition_keeps_every_original_sentence_and_order(self):
        for task in ['informativeness','humanitarian']:
            original=prompts.get_prompt_set(task)
            image,text=prompts.get_branch_prompt_sets(task)
            self.assertEqual(list(image),list(original))
            self.assertEqual(list(text),list(original))
            for key in original:
                self.assertEqual(text[key]+image[key],original[key])
                self.assertTrue(all('tweet' in sentence for sentence in text[key]))
                self.assertTrue(all('tweet' not in sentence for sentence in image[key]))
            image[next(iter(image))].append('mutation')
            self.assertNotIn('mutation',original[next(iter(original))])

    def test_encoder_receives_distinct_banks_with_same_aggregation(self):
        image,text=prompts.get_branch_prompt_sets('humanitarian')
        labels=list(text)
        with patch('run.run_crisismmd.build_class_embeddings',side_effect=[(np.ones((8,2)),labels),(np.zeros((8,2)),labels)]) as build:
            t,order,i,meta,audit=prepare_prompt_embeddings(None,{'task':'humanitarian','prompt_variant':'modality_split','prompt_aggregation':'mean_embed'})
        self.assertEqual(build.call_args_list[0].args[1],text)
        self.assertEqual(build.call_args_list[1].args[1],image)
        self.assertTrue(all(c.kwargs['aggregation']=='mean_embed' for c in build.call_args_list))
        self.assertEqual(audit,{'image':image,'text':text})
        self.assertTrue((t==1).all() and (i==0).all())

    def test_override_cannot_silently_replace_split(self):
        with self.assertRaises(ValueError):
            prepare_prompt_embeddings(None,{'task':'humanitarian','prompt_variant':'modality_split','_research_image_prompts':{}})

    def test_trial_expansion_and_distinct_result_path(self):
        trials=prompt_trial_configs('crisismmd',{'task':'informativeness','prompt_variants':['current','modality_split'],'prompt_aggregation':'mean_embed'})
        self.assertEqual(len(trials),2)
        job={'kind':'standard','dataset':'crisismmd','task':'informativeness','max_samples':5,
             'models':['metaclip2_vith14'],'prompt_variants':['modality_split']}
        self.assertEqual(job_path(job,'abc').parts[1],'modality_split')
        with self.assertRaises(ValueError):
            prompt_trial_configs('mmimdb',{'task':'genres','prompt_variants':['modality_split']})


if __name__=='__main__': unittest.main()
