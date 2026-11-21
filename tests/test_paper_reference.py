"""Kiểm thử tham chiếu chỉ dùng vector tổng hợp; không tải weights hoặc ảnh."""
import contextlib
import io as stdio
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vlm_project import io, kaggle, paper_reference as reference, protocol


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = io.config()

    def prepare(self):
        launcher = yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        original = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda p: self.root/'launches'
                          if str(p)=='results/_system/launches/final' else original(p)):
            return kaggle.prepare(launcher, 'paper_reference_dtd_vitl14')

    def test_reference_launcher_uses_separate_kernel_and_keeps_main_model(self):
        original = (ROOT/'configs/project.yaml').read_bytes()
        plan = self.prepare()
        self.assertEqual(plan['task'], 'paper_reference')
        self.assertEqual(plan['samples'], 1880)
        self.assertTrue(plan['kernel'].endswith('dtd-openai-vitl14-paper'))
        self.assertEqual(plan['reference_model'], 'openai_clip_vitl14')
        self.assertEqual((ROOT/'configs/project.yaml').read_bytes(), original)
        document = io.read_json(Path(plan['root'])/'kernel/run.ipynb')
        code = '\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type']=='code')
        self.assertIn('from vlm_project.paper_reference import run', code)
        self.assertNotIn('from vlm_project.benchmark import run', code)
        self.assertNotIn('freeze(settings', code)
        for cell in document['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']), '<reference>', 'exec')

    def test_imagenetv2_reference_launcher_selects_full_10000_and_correct_mount(self):
        launcher = yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        original = kaggle.path
        with patch.object(kaggle,'path',side_effect=lambda p:self.root/'launches'
                          if str(p)=='results/_system/launches/final' else original(p)):
            plan = kaggle.prepare(launcher,'paper_reference_imagenetv2_vitl14')
        self.assertEqual(plan['samples'],10000)
        self.assertEqual(plan['name'],'imagenetv2')
        self.assertEqual(plan['data_dataset'],'nguyenhung3105/vlm-public-benchmarks-eval-v1')
        self.assertTrue(plan['kernel'].endswith('imagenetv2-openai-vitl14-paper'))
        notebook=io.read_json(Path(plan['root'])/'kernel/run.ipynb')
        code='\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
        self.assertIn("name = 'imagenetv2'",code)
        self.assertNotIn('name = "dtd"',code)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'<imagenetv2>', 'exec')

    def test_reference_targets_pick_imagenetv2_row_separately_from_dtd(self):
        shutil.copytree(ROOT/'configs/paper_results', self.root/'configs/paper_results')
        with patch.object(reference,'ROOT',self.root):
            targets=reference.reference_targets('imagenetv2')
        self.assertAlmostEqual(targets['clip_photo']['author_csv_accuracy_percent'],67.90000200271606)
        self.assertAlmostEqual(targets['dclip_photo']['author_csv_accuracy_percent'],69.72000002861023)
        self.assertAlmostEqual(targets['waffle']['author_csv_accuracy_percent'],69.48428494589669)
        self.assertEqual(targets['dclip_original']['paper_accuracy_percent'],69.30)
        self.assertFalse((self.root/'third_party').exists())

    def test_reference_trials_keep_both_dclip_protocols_and_seven_waffle_seeds(self):
        def generate(api, mode, names, descriptor, pairs, seed):
            return {name:[mode+' '+name] for name in names}, None
        with patch.object(reference, 'generate_prompts', side_effect=generate):
            from types import SimpleNamespace
            api = SimpleNamespace(clip=SimpleNamespace(tokenize=lambda values: None))
            trials = reference.reference_trials(api, ['one','two'], 'unused')
        self.assertEqual(len(trials),11)
        self.assertEqual([t['seed'] for t in trials if t['method']=='waffle'], list(range(7)))
        photo = next(t for t in trials if t['method']=='dclip_photo')['prompts']
        original = next(t for t in trials if t['method']=='dclip_original')['prompts']
        self.assertNotEqual(photo, original)

    def test_reference_full_pipeline_saves_replay_and_paper_deltas(self):
        names = [f'c{i}' for i in range(47)]
        rows = [{'id':str(i),'label':i%47,'image':'synthetic'} for i in range(1880)]
        source = self.root/'source'
        descriptor = source/'third_party/WaffleCLIP/descriptors/descriptors_dtd.json'
        io.write_json(descriptor,{name:['detail'] for name in names})
        (source/'configs').mkdir()
        (source/'configs/project.yaml').write_text('model: metaclip2_vith14\n')
        trials=[]
        for method in ['clip_photo','dclip_photo','waffle','clip_labels_original','dclip_original']:
            for seed in list(range(7)) if method=='waffle' else [None]:
                trials.append({'method':method,'seed':seed,'prompts':{n:[n] for n in names}})
        class Fake:
            calls=0
            def __init__(self,device,api):self.metadata={'synthetic':True}
            def encode_images(self,rows):
                Fake.calls+=1
                return np.arange(len(rows))
            def classify(self,features,trial):
                pred=features%47
                top5=np.stack([(pred+k)%47 for k in range(5)],axis=1)
                probabilities=np.zeros((len(pred),47),dtype=np.float16)
                probabilities[np.arange(len(pred)),pred]=1
                return pred,top5,probabilities
            def unload(self):pass
        original = reference.path
        targets = {m:{'paper_accuracy_percent':50.} for m in {t['method'] for t in trials}}
        with patch.object(reference,'ROOT',source),patch.object(reference,'verify_sources',return_value={}),\
             patch.object(reference,'author_api',return_value=None),patch.object(reference,'reference_trials',return_value=trials),\
             patch.object(reference,'reference_targets',return_value=targets),\
             patch.object(reference.data,'benchmark',return_value=(names,rows,{})),\
             patch.object(reference,'path',side_effect=lambda p:self.root/'output' if str(p)=='results/final/paper_reference/dtd' else original(p)),\
             contextlib.redirect_stdout(stdio.StringIO()):
            output=reference.run(self.settings,'dtd','synthetic',backend_factory=Fake)
        manifest=protocol.completed_run(output)
        self.assertEqual(Fake.calls,1)
        self.assertEqual(manifest['model'],'openai_clip_vitl14')
        self.assertEqual(manifest['settings']['model'],'openai_clip_vitl14')
        self.assertEqual(len(manifest['summary']),5)
        with np.load(output/'replay_scores.npz',allow_pickle=False) as saved:
            self.assertEqual(saved['labels'].shape,(1880,))
            self.assertEqual(saved['waffle_seed0_top5'].shape,(1880,5))
        compared=io.read_json(output/'paper_comparison.json')
        self.assertEqual(len(compared),5)
        self.assertFalse(any(r['approx_match_within_0_10pp'] for r in compared))

    def test_reference_submission_pins_returned_version(self):
        plan=self.prepare()
        def cli(arguments,check=True):
            if arguments[:2]==['kernels','status']:
                return subprocess.CompletedProcess(arguments,1,'','404 not found')
            if arguments[:2]==['kernels','push']:
                return subprocess.CompletedProcess(arguments,0,'Kernel version 3 successfully pushed.','')
            return subprocess.CompletedProcess(arguments,0,'{"status":"ready"}','')
        with patch.object(kaggle,'cli',side_effect=cli),contextlib.redirect_stdout(stdio.StringIO()):
            result=kaggle.submit(plan)
        self.assertEqual(result['kernel_reference'],plan['kernel']+'/3')

    def test_reference_does_not_run_other_dataset(self):
        with self.assertRaisesRegex(ValueError,'DTD'):
            reference.run(self.settings,'unknown','unused')

    def test_imagenet_reference_selects_full_validation_and_distinct_kernel(self):
        launcher = yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        original = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda p:self.root/'launches'
                          if str(p)=='results/_system/launches/final' else original(p)):
            plan = kaggle.prepare(launcher, 'paper_reference_imagenet_vitl14')
        self.assertEqual(plan['samples'], 50000)
        self.assertEqual(plan['name'], 'imagenet')
        self.assertTrue(plan['kernel'].endswith('/vlm-final-imagenet-openai-vitl14-paper'))
        document = io.read_json(Path(plan['root'])/'kernel/run.ipynb')
        code = '\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type']=='code')
        self.assertIn("name = 'imagenet'", code)
        for cell in document['cells']:
            if cell['cell_type']=='code': compile(''.join(cell['source']), '<imagenet>', 'exec')

    def test_imagenet_original_controls_follow_author_period_suffix(self):
        from types import SimpleNamespace
        api = SimpleNamespace(clip=SimpleNamespace(tokenize=lambda values: None))
        def generate(api, mode, names, descriptor, pairs, seed):
            return {name:[mode+' '+name] for name in names}, None
        with patch.object(reference, 'generate_prompts', side_effect=generate):
            base = reference.reference_trials(api, ['one','two'], 'unused', dataset='imagenetv2')
            imagenet = reference.reference_trials(api, ['one','two'], 'unused', dataset='imagenet')
        for before, after in zip(base, imagenet):
            suffix = '.' if before['method'] in ['clip_labels_original','dclip_original'] else ''
            self.assertEqual(after['prompts'], {n:[v+suffix for v in values]
                                              for n,values in before['prompts'].items()})

    def test_cub_reference_selects_full_test_and_correct_prompts(self):
        launcher = yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        original = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda p:self.root/'launches'
                          if str(p)=='results/_system/launches/final' else original(p)):
            plan = kaggle.prepare(launcher, 'paper_reference_cub_vitl14')
        self.assertEqual(plan['samples'],5794)
        self.assertEqual(plan['name'],'cub')
        self.assertTrue(plan['kernel'].endswith('/vlm-final-cub-openai-vitl14-paper'))
        self.assertEqual(plan['data_dataset'],'nguyenhung3105/vlm-public-benchmarks-eval-v1')
        document = io.read_json(Path(plan['root'])/'kernel/run.ipynb')
        code = '\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type']=='code')
        self.assertIn("name = 'cub'",code)
        for cell in document['cells']:
            if cell['cell_type']=='code': compile(''.join(cell['source']),'<cub>','exec')
        from types import SimpleNamespace
        api = SimpleNamespace(clip=SimpleNamespace(tokenize=lambda values: None))
        def generate(api, mode, names, descriptor, pairs, seed):
            return {name:[mode+' '+name] for name in names},None
        with patch.object(reference,'generate_prompts',side_effect=generate):
            trials = reference.reference_trials(api,['bird'],'unused',dataset='cub')
        self.assertEqual(len(trials),11)
        controls = [t for t in trials if t['method'] in ['dclip_original','clip_labels_original']]
        self.assertTrue(all(not t['prompts']['bird'][0].endswith('.') for t in controls))

    def test_food101_reference_selects_full_test_and_original_prompt_controls(self):
        launcher = yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        original = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda p:self.root/'launches'
                          if str(p)=='results/_system/launches/final' else original(p)):
            plan = kaggle.prepare(launcher,'paper_reference_food101_vitl14')
        self.assertEqual(plan['samples'],25250)
        self.assertEqual(plan['name'],'food101')
        self.assertEqual(reference.REFERENCE_DATASETS['food101'],{'descriptor':'food101','split':'test'})
        self.assertTrue(plan['kernel'].endswith('/vlm-final-food101-openai-vitl14-paper'))
        self.assertEqual(plan['data_dataset'],'nguyenhung3105/vlm-public-benchmarks-eval-v1')
        document = io.read_json(Path(plan['root'])/'kernel/run.ipynb')
        code = '\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type']=='code')
        self.assertIn("name = 'food101'",code)
        for cell in document['cells']:
            if cell['cell_type']=='code': compile(''.join(cell['source']),'<food101>','exec')
        from types import SimpleNamespace
        api = SimpleNamespace(clip=SimpleNamespace(tokenize=lambda values:None))
        def generate(api,mode,names,descriptor,pairs,seed):
            return {name:[mode+' '+name] for name in names},None
        with patch.object(reference,'generate_prompts',side_effect=generate):
            trials = reference.reference_trials(api,['apple pie'],'unused',dataset='food101')
        self.assertEqual(len(trials),11)
        self.assertEqual([t['seed'] for t in trials if t['method']=='waffle'],list(range(7)))
        controls = [t for t in trials if t['method'] in ['dclip_original','clip_labels_original']]
        self.assertTrue(all(not t['prompts']['apple pie'][0].endswith('.') for t in controls))

    def test_remaining_reference_launchers_select_full_dataset_and_separate_kernel(self):
        launcher = yaml.safe_load((ROOT/'configs/kaggle_launcher.yaml').read_text())
        original = kaggle.path
        for dataset, samples in [('pets',3669),('eurosat',27000),('places365',36500)]:
            with self.subTest(dataset=dataset), patch.object(kaggle,'path',side_effect=lambda p:self.root/'launches'
                        if str(p)=='results/_system/launches/final' else original(p)):
                plan = kaggle.prepare(launcher,f'paper_reference_{dataset}_vitl14')
                self.assertEqual(plan['samples'],samples)
                self.assertEqual(plan['name'],dataset)
                self.assertTrue(plan['kernel'].endswith(f'/vlm-final-{dataset}-openai-vitl14-paper'))
                self.assertEqual(plan['data_dataset'],'nguyenhung3105/vlm-public-benchmarks-eval-v1')
                self.assertEqual(reference.REFERENCE_DATASETS[dataset]['split'],io.BENCHMARKS[dataset][2])
                document = io.read_json(Path(plan['root'])/'kernel/run.ipynb')
                code = '\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type']=='code')
                self.assertIn(f'name = {dataset!r}',code)
                for cell in document['cells']:
                    if cell['cell_type']=='code':compile(''.join(cell['source']),f'<{dataset}>','exec')

    def test_pets_column_alignment_preserves_author_rng_and_all_prompt_strings(self):
        from types import SimpleNamespace
        api = SimpleNamespace(clip=SimpleNamespace(tokenize=lambda values:None))
        def generate(api,mode,names,descriptor,pairs,seed):
            # Phụ thuộc cả vị trí lớp và seed, để phát hiện sinh lại sai thứ tự.
            return {name:[f'{mode} {index} {seed} {name}'] for index,name in enumerate(names)},None
        author_names = ['cat','dog','bird']; dataset_names = ['bird','cat','dog']
        with patch.object(reference,'generate_prompts',side_effect=generate):
            original = reference.reference_trials(api,author_names,'unused',dataset='pets')
            aligned = reference.reference_trials(api,author_names,'unused',dataset='pets',label_order=dataset_names)
        self.assertEqual(len(aligned),11)
        for before,after in zip(original,aligned):
            self.assertEqual(list(after['prompts']),dataset_names)
            self.assertEqual(after['prompts'],before['prompts'])
            for name in author_names:self.assertEqual(after['prompts'][name],before['prompts'][name])

    def test_reference_alignment_rejects_missing_duplicate_or_renamed_classes(self):
        for labels in [['cat','dog'],['cat','cat','bird'],['cat','dog','Bird']]:
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError,'Taxonomy'):
                reference.reference_trials(None,['cat','dog','bird'],'unused',dataset='pets',label_order=labels)

    def test_pets_pipeline_preserves_dataset_truth_and_records_author_mapping(self):
        from types import SimpleNamespace
        names = [f'breed{i:02}' for i in range(37)]
        author_names = list(reversed(names))
        rows = [{'id':str(i),'label':i,'image':'synthetic'} for i in range(5)]
        io.write_json(self.root/'third_party/WaffleCLIP/descriptors/descriptors_pets.json',
                      {name:['detail'] for name in author_names})
        (self.root/'configs').mkdir()
        (self.root/'configs/project.yaml').write_text('model: metaclip2_vith14\n')
        api = SimpleNamespace(clip=SimpleNamespace(tokenize=lambda values:None))
        def generate(api,mode,labels,descriptor,pairs,seed):
            self.assertEqual(labels,author_names)
            return {name:[f'{index} {name}'] for index,name in enumerate(labels)},None
        check = self
        class Fake:
            def __init__(self,device,api):self.metadata={'synthetic':True}
            def encode_images(self,rows):return np.arange(5)
            def classify(self,features,trial):
                check.assertEqual(list(trial['prompts']),names)
                for name in names:
                    check.assertEqual(trial['prompts'][name],[f'{author_names.index(name)} {name}'])
                probabilities=np.eye(37,dtype=np.float16)[features]
                return features,np.stack([(features+k)%37 for k in range(5)],axis=1),probabilities
            def unload(self):pass
        original = reference.path
        with patch.object(reference,'ROOT',self.root), patch.object(reference,'verify_sources',return_value={}),\
             patch.object(reference,'author_api',return_value=api),\
             patch.object(reference,'generate_prompts',side_effect=generate),\
             patch.object(reference,'reference_targets',return_value={}),\
             patch.object(reference.data,'benchmark',return_value=(names,rows,{})),\
             patch.object(reference,'path',side_effect=lambda p:self.root/'output'
                          if str(p)=='results/final/paper_reference/pets' else original(p)),\
             contextlib.redirect_stdout(stdio.StringIO()):
            output=reference.run(self.settings,'pets','synthetic',smoke=True,backend_factory=Fake)
        manifest=protocol.completed_run(output)
        self.assertEqual(manifest['classes'],names)
        self.assertEqual(manifest['author_class_order'],author_names)
        self.assertEqual(manifest['author_to_dataset_label'],list(reversed(range(37))))
        self.assertTrue(all(row['accuracy_mean']==1. for row in manifest['summary']))
        self.assertEqual([row['label'] for row in io.read_json(output/'cohort.json')],list(range(5)))

    def test_imagenet_targets_do_not_pick_imagenetv2_rows(self):
        shutil.copytree(ROOT/'configs/paper_results', self.root/'configs/paper_results')
        with patch.object(reference, 'ROOT', self.root):
            targets = reference.reference_targets('imagenet')
        self.assertAlmostEqual(targets['clip_photo']['author_csv_accuracy_percent'],73.3739972114563)
        self.assertAlmostEqual(targets['dclip_photo']['author_csv_accuracy_percent'],75.2560019493103)
        self.assertAlmostEqual(targets['waffle']['author_csv_accuracy_percent'],75.30342766216823)
        self.assertEqual(targets['dclip_original']['paper_accuracy_percent'],75.00)

    def test_all_saved_targets_match_pinned_csv_and_preserve_paper_disagreement(self):
        import csv
        registry = io.read_json(ROOT/'configs/paper_results/references.json')
        self.assertEqual(set(registry['datasets']), set(io.BENCHMARKS))
        for name in registry['datasets']:
            targets = reference.reference_targets(name)
            for method in ['clip_photo','dclip_photo','waffle']:
                target = targets[method]
                source = ROOT/registry['author_csv_snapshots'][target['source_file']]['file']
                self.assertEqual(io.sha256(source), target['source_sha256'])
                with source.open(newline='') as stream:
                    rows = [r for r in csv.reader(stream) if
                            f'dataset={name};' in r[0] and 'model_size=ViT-L/14' in r[0]]
                self.assertEqual(len(rows),1)
                self.assertEqual(target['author_csv_accuracy_percent'],float(rows[0][1]))
                self.assertEqual(target['author_csv_std_percent_ddof0'],float(rows[0][2]))
        cub = reference.reference_targets('cub')['clip_photo']
        self.assertEqual(cub['paper_accuracy_percent'],62.24)
        self.assertAlmostEqual(cub['author_csv_accuracy_percent'],62.27131485939026)
        self.assertEqual(registry['datasets']['imagenet']['clip']['paper_score_percent'],75.3)
        self.assertIsNone(registry['datasets']['imagenetv2']['clip']['paper_score_percent'])
        self.assertEqual(registry['datasets']['pets']['clip']['metric'],'mean_per_class_accuracy')

    def test_saved_csv_tampering_stops_comparison(self):
        shutil.copytree(ROOT/'configs/paper_results', self.root/'configs/paper_results')
        source = self.root/'configs/paper_results/sources/baselines.csv'
        source.write_text(source.read_text()+'\n')
        with patch.object(reference, 'ROOT', self.root), self.assertRaisesRegex(ValueError,'checksum'):
            reference.reference_targets('dtd')

    def test_packaged_reference_targets_load_without_third_party_or_network(self):
        import zipfile
        from vlm_project.packaging import package
        receipt = package(self.root/'bundle')
        with zipfile.ZipFile(receipt['zip']) as archive:
            archive.extractall(self.root/'unpacked')
        with patch.object(reference, 'ROOT', self.root/'unpacked'):
            targets = reference.reference_targets('dtd')
        self.assertEqual(targets['clip_photo']['paper_accuracy_percent'],52.87)
        self.assertEqual(targets['dclip_original']['paper_accuracy_percent'],54.36)
        self.assertIn('reference_registry_sha256', targets['waffle'])

    def test_uploaded_code_status_403_can_use_verified_zip_file_list(self):
        plan=self.prepare()
        source=Path(plan['root'])/'code/final_model_source.zip'
        responses=[subprocess.CompletedProcess([],1,'','403 Forbidden'),
                   subprocess.CompletedProcess([],0,json.dumps([{'name':source.name,'size':source.stat().st_size}]),'')]
        with patch.object(kaggle,'cli',side_effect=responses):
            self.assertTrue(kaggle.code_dataset_ready(plan))

    def test_uploaded_code_status_403_rejects_incomplete_or_wrong_files(self):
        plan=self.prepare()
        for rows in [[],[{'name':'final_model_source.zip','size':1}]]:
            responses=[subprocess.CompletedProcess([],1,'','403 Forbidden'),
                       subprocess.CompletedProcess([],0,json.dumps(rows),'')]
            with patch.object(kaggle,'cli',side_effect=responses):
                self.assertFalse(kaggle.code_dataset_ready(plan))

    def test_uploaded_code_status_403_accepts_complete_extracted_manifest(self):
        import zipfile
        plan=self.prepare()
        with zipfile.ZipFile(Path(plan['root'])/'code/final_model_source.zip') as source:
            rows=[{'name':'final_model_source/'+entry.filename,'size':entry.file_size}
                  for entry in source.infolist() if not entry.is_dir()]
        responses=[subprocess.CompletedProcess([],1,'','403 Forbidden'),
                   subprocess.CompletedProcess([],0,json.dumps(rows),'')]
        with patch.object(kaggle,'cli',side_effect=responses):
            self.assertTrue(kaggle.code_dataset_ready(plan))


if __name__=='__main__':unittest.main()
