"""Protocol/runner test hai tầng dùng artifact và encoder tổng hợp, không mở test thật."""
import contextlib
import copy
import importlib.util
import io as stdio
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import io, protocol, fakeddit_fewshot_cascade as cascade
from vlm_project import cascade_protocol as cp, cascade_test_runner as runner
import test_fakeddit_fewshot_cascade as fixtures
import test_min5_test_runner as backends

RUNTIME = {'python': '3.11.11', 'libraries': {
    'numpy': '2.1.0', 'pillow': '11.1.0', 'pyyaml': '6.0.2', 'scipy': '1.15.0',
    'scikit-learn': '1.6.1', 'torch': '2.6.0', 'torchvision': '0.21.0',
    'open_clip_torch': '3.3.0', 'transformers': '4.55.0', 'sentencepiece': '0.2.0',
    'protobuf': '5.29.0', 'ftfy': '6.3.0', 'regex': '2024.11.6', 'packaging': '25.0', 'tqdm': '4.67.0'}}


class CascadeTestProtocolTests(unittest.TestCase):
    def fixture(self, root, sealed=False):
        snapshot, settings, roots, rows, load = fixtures.CascadeTests().fixture(root)
        with patch.object(cascade.data, 'fakeddit', side_effect=load), \
             patch.object(cascade, 'encode_cohort', side_effect=fixtures.CascadeTests.encode), \
             contextlib.redirect_stdout(stdio.StringIO()):
            dev = cascade.run(settings, roots['train'], roots['dev'], backend_factory=backends.Backend, spec_file=snapshot)
        settings_file = root / 'settings.yaml'
        draft = root / 'draft_cascade.json'
        cp.prepare(dev, root / 'source', draft, settings_file, spec_file=snapshot)
        runtime_file = root / 'runtime.json'
        with patch.object(cp, 'runtime_versions', return_value=copy.deepcopy(RUNTIME)):
            cp.capture_runtime(runtime_file)
        if sealed:
            cp.seal(draft, io.sha256(roots['test'] / 'manifest.json'), runtime_file, root / 'sealed_cascade.json')
        test = [{'id': f'new{i}', '6way': str(i % 6), 'image': 'unused', 'text': 'unused',
                 'image_sha256': io.digest(['new', i]), 'text_sha256': io.digest(['new text', i])} for i in range(3000)]
        rows['test'] = test
        def test_loader(directory, split):
            self.assertIn(split, ['train','test'])
            return rows[split], {'split': split, 'source_split': 'train' if split == 'train' else 'test_public',
                                 'manifest_sha256': io.sha256(roots[split] / 'manifest.json')}
        return draft, runtime_file, roots, rows, test_loader, settings_file, snapshot, dev

    def test_draft_preserves_six_methods_support_thresholds_and_is_not_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); draft, runtime_file, roots, rows, load, settings, snapshot, dev = self.fixture(root)
            lock = cp.validate(draft, require_sealed=False)
            p = lock['payload']
            self.assertEqual(p['state'], 'draft')
            self.assertEqual(p['stages'], 2); self.assertEqual(len(p['methods']), 6)
            self.assertEqual(p['recipe'], cascade.specification(snapshot))
            self.assertEqual(p['primary_metric'], 'macro_f1')
            self.assertEqual(len(p['excluded_train']), 3000); self.assertEqual(len(p['excluded_dev']), 3000)
            self.assertFalse(runner.plan(draft)['executable'])
            with self.assertRaisesRegex(ValueError, 'draft'): cp.validate(draft)
            with self.assertRaises(FileExistsError): cp.prepare(dev, root/'source', draft, settings, spec_file=snapshot)
            with patch.object(runner.data, 'fakeddit', side_effect=AssertionError('dataset accessed')):
                with self.assertRaises(ValueError): runner.run(draft, roots['train'], roots['test'])

    def test_seal_requires_test_hash_target_runtime_and_preserves_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);draft,rt,roots,rows,load,settings,snapshot,dev=self.fixture(root)
            for bad in ['', 'a'*63, 'v2', io.sha256(roots['train']/'manifest.json'), io.sha256(roots['dev']/'manifest.json')]:
                with self.assertRaises(ValueError): cp.seal(draft,bad,rt,root/'invalid.json')
            cp.seal(draft,io.sha256(roots['test']/'manifest.json'),rt,root/'sealed_target.json')
            with patch.object(cp,'runtime_versions',return_value=copy.deepcopy(RUNTIME)):
                p=cp.validate(root/'sealed_target.json')['payload']
                self.assertEqual(p['recipe'],cascade.specification(snapshot))
                self.assertEqual(p['runtime'],RUNTIME)
                self.assertTrue(runner.plan(root/'sealed_target.json')['executable'])
            with self.assertRaises(FileExistsError): cp.seal(draft,'b'*64,rt,root/'sealed_target.json')
            changed=copy.deepcopy(RUNTIME);changed['libraries']['numpy']='2.2.0'
            with patch.object(cp,'runtime_versions',return_value=changed):
                with self.assertRaisesRegex(ValueError,'runtime'): cp.validate(root/'sealed_target.json')
                self.assertFalse(runner.plan(root/'sealed_target.json')['executable'])

    def test_code_config_snapshot_and_payload_tampering_fail_before_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);draft,rt,roots,rows,load,settings,snapshot,dev=self.fixture(root,sealed=True)
            sealed=root/'sealed_cascade.json'
            with patch.object(cp,'runtime_versions',return_value=RUNTIME):
                for case in ['digest','kind','state','test_opened','suite','threshold','valid_threshold','excluded']:
                    with self.subTest(case=case):
                        lock=io.read_json(sealed);p=lock['payload']
                        if case=='digest':lock['sha256']='bad'
                        if case=='kind':p['kind']='min5_fewshot_task1_protocol'
                        if case=='state':p['state']='sealed_task1'
                        if case=='test_opened':p['test_opened']=True
                        if case=='suite':p['methods'].pop()
                        if case=='threshold':p['recipe']['gate_candidates'][1]['threshold']=2.
                        if case=='valid_threshold':
                            p['recipe']['gate_candidates'][1]['threshold']=.1;p['recipe_sha256']=io.digest(p['recipe'])
                        if case=='excluded':p['excluded_dev'].pop()
                        if case!='digest':lock['sha256']=io.digest(p)
                        io.write_json(root/'bad.json',lock)
                        with patch.object(runner.data,'fakeddit',side_effect=AssertionError('dataset accessed')):
                            with self.assertRaises(ValueError): runner.run(root/'bad.json',roots['train'],roots['test'])
                with patch.object(cp,'fingerprint',return_value={}):
                    with self.assertRaises(ValueError):cp.validate(sealed)
                settings.write_text('model: synthetic\nbatch_size: 16\n')
                with self.assertRaises(ValueError):cp.validate(sealed)

    def test_full_test_inference_has_six_methods_no_dev_partition_and_fixed_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);draft,rt,roots,rows,load,settings,snapshot,dev=self.fixture(root,sealed=True)
            backend=backends.Backend('cpu')
            with patch.object(cp,'runtime_versions',return_value=RUNTIME), \
                 patch.object(runner.data,'fakeddit',side_effect=load), \
                 patch.object(runner,'encode_cohort',side_effect=fixtures.CascadeTests.encode) as encode, \
                 contextlib.redirect_stdout(stdio.StringIO()):
                output=runner.run(root/'sealed_cascade.json',roots['train'],roots['test'],root/'test_output',backend_factory=lambda _:backend)
            self.assertTrue(backend.unloaded);self.assertEqual(encode.call_count,2)
            r=protocol.completed_run(output)
            self.assertEqual((r['stages'],r['split'],r['test_opened'],r['samples']),(2,'test',True,3000))
            self.assertEqual(len(r['summary']),6);self.assertFalse((output/'audit_summary.tsv').exists())
            self.assertEqual(io.read_json(output/'support.json'),cascade.specification(snapshot)['support_snapshot'])
            audit_spec=importlib.util.spec_from_file_location('test_cascade_audit',ROOT/'scripts/audit_report_results.py')
            audit=importlib.util.module_from_spec(audit_spec);audit_spec.loader.exec_module(audit);audit.ROOT=root.resolve()
            self.assertEqual(audit.audit(output)['prediction_rows'],90000)

    def test_test_labels_do_not_change_model_scores_or_hard_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);draft,rt,roots,rows,load,settings,snapshot,dev=self.fixture(root,sealed=True)
            with patch.object(cp,'runtime_versions',return_value=RUNTIME), \
                 patch.object(runner.data,'fakeddit',side_effect=load), \
                 patch.object(runner,'encode_cohort',side_effect=fixtures.CascadeTests.encode), \
                 contextlib.redirect_stdout(stdio.StringIO()):
                a=runner.run(root/'sealed_cascade.json',roots['train'],roots['test'],root/'outputs',backend_factory=backends.Backend)
                for r in rows['test']:r['6way']=str((int(r['6way'])+1)%6)
                b=runner.run(root/'sealed_cascade.json',roots['train'],roots['test'],root/'outputs',backend_factory=backends.Backend)
            with np.load(a/'replay_scores.npz',allow_pickle=False) as x,np.load(b/'replay_scores.npz',allow_pickle=False) as y:
                self.assertFalse(np.array_equal(x['labels_6way'],y['labels_6way']))
                for key in x.files:
                    if key.startswith(('p_','prediction_')):np.testing.assert_array_equal(x[key],y[key])

    def test_wrong_manifest_overlap_and_runtime_mismatch_stop_before_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);draft,rt,roots,rows,load,settings,snapshot,dev=self.fixture(root,sealed=True)
            with patch.object(cp,'runtime_versions',return_value=RUNTIME):
                with patch.object(runner.data,'fakeddit',side_effect=AssertionError('dataset accessed')):
                    with self.assertRaises(ValueError):runner.run(root/'sealed_cascade.json',roots['test'],roots['train'])
                for excluded in ['train','dev']:
                    for key in ['id','image_sha256','text_sha256']:
                        old=rows['test'][0][key];rows['test'][0][key]=rows[excluded][0][key]
                        with patch.object(runner.data,'fakeddit',side_effect=load):
                            with self.assertRaisesRegex(ValueError,'overlap'):
                                runner.run(root/'sealed_cascade.json',roots['train'],roots['test'],backend_factory=self.fail)
                        rows['test'][0][key]=old

    def test_unload_when_encoder_fails_and_prepare_only_never_reads_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);draft,rt,roots,rows,load,settings,snapshot,dev=self.fixture(root,sealed=True)
            backend=backends.Backend('cpu')
            with patch.object(cp,'runtime_versions',return_value=RUNTIME), \
                 patch.object(runner.data,'fakeddit',side_effect=load), \
                 patch.object(runner,'encode_cohort',side_effect=RuntimeError('encoding')):
                with self.assertRaises(RuntimeError):runner.run(root/'sealed_cascade.json',roots['train'],roots['test'],root/'outputs',backend_factory=lambda _:backend)
            self.assertTrue(backend.unloaded)
            p=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_cascade_test.py'),'--protocol',str(draft),
                              '--prepare-only','--test-root','/does/not/exist'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertIn('"executable": false',p.stdout)

    def test_bad_runtime_capture_and_smoke_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);draft,rt,roots,rows,load,settings,snapshot,dev=self.fixture(root)
            for case in ['missing','requirement','python']:
                value=copy.deepcopy(RUNTIME)
                if case=='missing':value['libraries']['torch']=None
                if case=='requirement':value['libraries']['open_clip_torch']='3.2.0'
                if case=='python':value['python']=None
                with patch.object(cp,'runtime_versions',return_value=value):
                    with self.assertRaises(ValueError):cp.capture_runtime(root/'bad_runtime.json')
            run=io.read_json(dev/'run.json');run['mode']='smoke'
            io.write_json(dev/'run.json',run)
            io.write_json(dev/'completed.json',{'status':'complete','run_sha256':io.sha256(dev/'run.json')})
            with self.assertRaises(ValueError):cp.prepare(dev,root/'source',root/'invalid.json',settings,spec_file=snapshot)

    def test_relocated_configuration_and_model_metadata_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);draft,rt,roots,rows,load,settings,snapshot,dev=self.fixture(root,sealed=True)
            replacement=root/'relocated.yaml';replacement.write_bytes(settings.read_bytes());settings.unlink()
            backend=backends.Backend('cpu');backend.metadata['pretrained']='wrong'
            with patch.object(cp,'runtime_versions',return_value=RUNTIME), \
                 patch.object(runner.data,'fakeddit',side_effect=load), \
                 patch.object(runner,'encode_cohort',side_effect=AssertionError('encoder called')):
                with self.assertRaises(ValueError):
                    runner.run(root/'sealed_cascade.json',roots['train'],roots['test'],root/'outputs',replacement,backend_factory=lambda _:backend)
            self.assertTrue(backend.unloaded)


if __name__=='__main__':unittest.main()
