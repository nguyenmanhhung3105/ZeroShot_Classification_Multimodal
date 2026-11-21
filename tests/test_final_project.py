"""Chỉ dữ liệu/vector tổng hợp; không tải trọng số hay đọc dataset thật."""
import contextlib
import csv
import hashlib
import io as stdio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import data, benchmark, fakeddit, protocol
from vlm_project import io, evaluate, packaging
from vlm_project.methods import hierarchy, replace_clip_vote
from vlm_project.backbone import Backbone
from vlm_project import kaggle

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

class FakeBackend:
    """Backend test CPU; encode_cohort được mock tường minh, không đọc ảnh."""
    calls = 0

    def __init__(self, device):
        self.device = device

    def prototypes(self, bank):
        result = []
        for prompts in bank.values():
            seed = int(hashlib.sha256(json.dumps(prompts).encode()).hexdigest()[:8], 16)
            result.append(unit(np.random.default_rng(seed).normal(size=(len(prompts), 12))).mean(0))
        return np.asarray(result, dtype=np.float32)

    def unload(self):
        pass

def fake_encode(backend, rows, batch_size, with_titles=False):
    FakeBackend.calls += 1
    rng = np.random.default_rng(17)
    features = unit(rng.normal(size=(len(rows), 12))).astype(np.float32)
    titles = unit(rng.normal(size=(len(rows), 12))).astype(np.float32) if with_titles else None
    return (features, titles)

class FinalProjectTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = io.config()
        self.settings['output_root'] = str(self.root / 'runs')
        self.settings['fakeddit']['lock_file'] = str(self.root / 'lock.json')

    def test_selected_predictions_match_legacy_fixture(self):
        fixture = json.loads((ROOT / 'tests/fixtures/legacy_predictions.json').read_text())
        features = np.asarray(fixture['features'], dtype=np.float32)
        titles = np.asarray(fixture['titles'], dtype=np.float32)
        means = {t: {b: np.asarray(v, dtype=np.float32) for b, v in banks.items()} for t, banks in fixture['means'].items()}
        text = {t: np.asarray(v, dtype=np.float32) for t, v in fixture['text_means'].items()}
        actual, diagnostics = fakeddit.predict(features, titles, means, text, fakeddit.profile(self.settings))
        for task, method, seed, predicted in actual:
            if method == 'proposed':
                np.testing.assert_array_equal(predicted, fixture['expected'][task][str(seed)])
        for seed in range(3):
            np.testing.assert_array_equal(diagnostics[f'groups_{seed}'], fixture['batch_ids'][str(seed)])

    def test_hierarchy_label_semantics_and_vote_replacement(self):
        result = hierarchy(np.array(['1', '0']), np.array([[0, 1, 2, 3, 4, 5], [0, 9, 2, 3, 4, 5]]), list('012345'))
        self.assertEqual(result.tolist(), ['0', '1'])
        pred, votes = replace_clip_vote([[0, 0, 1], [1, 1, 0]], [[0.1, 0.9], [0.9, 0.1]])
        self.assertEqual(pred.tolist(), [1, 0])
        self.assertEqual(votes.tolist(), [[1, 0, 1], [0, 1, 0]])

    def test_wrong_backbone_rejected(self):
        import yaml
        settings = io.config()
        settings['model'] = 'another_model'
        file = self.root / 'invalid.yaml'
        file.write_text(yaml.safe_dump(settings))
        with self.assertRaisesRegex(ValueError, 'MetaCLIP2'):
            io.config(file)

    def test_prompt_limit_uses_actual_worldwide_tokenizer(self):
        from types import SimpleNamespace
        calls = []

        def tokenizer(texts, **options):
            calls.append((texts, options))
            return {'input_ids': [[1] * int(text) for text in texts]}
        backend = Backbone.__new__(Backbone)
        backend.model = SimpleNamespace(context_length=77)
        backend.tokenizer = SimpleNamespace(tokenizer=tokenizer, clean_fn=str.strip)
        backend.check_prompts([' 77 '])
        with self.assertRaisesRegex(ValueError, 'context'):
            backend.check_prompts(['78'])
        self.assertFalse(calls[0][1]['truncation'])
        self.assertTrue(calls[0][1]['add_special_tokens'])

    def test_preserved_profile_checksum(self):
        migration = io.read_json(ROOT / 'configs/migration_sources.json')
        self.assertEqual(io.sha256(ROOT / 'configs/fakeddit_prompts.json'), migration['fakeddit_profile_sha256'])

    def test_forbidden_paths_and_symlink_escape(self):
        with self.assertRaises(ValueError):
            io.path(self.root / 'dataraw' / 'any')
        (self.root / 'link').symlink_to(self.root / 'data_process', target_is_directory=True)
        with self.assertRaises(ValueError):
            io.path(self.root / 'link' / 'any')

    def make_benchmark(self):
        root = self.root / 'dtd'
        (root / 'images').mkdir(parents=True)
        classes = [f'c{k}' for k in range(47)]
        io.write_json(root / 'classes.json', classes)
        io.write_json(root / 'provenance.json', {'dataset': 'dtd', 'split': 'test-partition-1'})
        rows = []
        for k in range(6):
            file = root / 'images' / f'{k}.jpg'
            file.write_bytes(f'fake-image-{k}'.encode())
            rows.append({'id': str(k), 'image': file.name, 'label': k, 'bytes': file.stat().st_size, 'sha256': io.sha256(file)})
        evaluate.table(root / 'manifest.tsv', rows)
        return (root, rows)

    def test_benchmark_loader_smoke_limit_corruption_and_full_count(self):
        root, rows = self.make_benchmark()
        (root / 'images/5.jpg').unlink()
        names, loaded, meta = data.benchmark(root, 'dtd', True)
        self.assertEqual(len(loaded), 5)
        self.assertEqual(len(names), 47)
        with self.assertRaises(FileNotFoundError):
            data.benchmark(root, 'dtd', False)
        (root / 'images/0.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            data.benchmark(root, 'dtd', True)

    def test_manifest_path_escape_rejected(self):
        root, rows = self.make_benchmark()
        rows[0]['image'] = '../../escape.jpg'
        evaluate.table(root / 'manifest.tsv', rows)
        with self.assertRaisesRegex(ValueError, 'path'):
            data.benchmark(root, 'dtd', True)

    def test_fakeddit_loader_rebases_paths_and_validates_title(self):
        root = self.root / 'fakeddit'
        (root / 'images').mkdir(parents=True)
        samples = []
        rows = []
        for k in range(3000):
            title = f'Synthetic title {k}'
            filename = f'{k}.jpg'
            checksum = hashlib.sha256(f'fake-image-{k}'.encode()).hexdigest()
            text_hash = hashlib.sha256(title.casefold().encode()).hexdigest()
            samples.append({'id': str(k), 'y_true': str(k % 6), 'image_sha256': checksum, 'text_sha256': text_hash})
            rows.append({'id': str(k), '6_way_label': str(k % 6), '2_way_label': str(int(k % 6 == 0)), 'clean_title': title, 'image_path': f'old/project/path/images/{filename}', 'split': 'dev', 'source_split': 'validate'})
            if k < 5:
                (root / 'images' / filename).write_bytes(f'fake-image-{k}'.encode())
        evaluate.table(root / 'fakeddit_sample.tsv', rows)
        manifest = {'status': 'ready', 'dataset': 'fakeddit', 'split': 'dev', 'source_split': 'validate', 'selected': {'samples': 3000}, 'samples': samples, 'cohort_id': io.digest(samples)[:20], 'samples_sha256': io.sha256(root / 'fakeddit_sample.tsv')}
        io.write_json(root / 'manifest.json', manifest)
        loaded, meta = data.fakeddit(root, 'dev', True)
        self.assertEqual(len(loaded), 5)
        self.assertEqual(Path(loaded[0]['image']).parent, (root / 'images').resolve())
        chosen, _ = data.fakeddit(root, 'dev', selected_indices=[1, 4])
        self.assertEqual([r['id'] for r in chosen], ['1', '4'])
        for indices in [[], [-1], [3000]]:
            with self.assertRaises(ValueError):
                data.fakeddit(root, 'dev', selected_indices=indices)
        with self.assertRaises(ValueError):
            data.fakeddit(root, 'test', selected_indices=[1])
        rows[0]['clean_title'] = 'changed'
        evaluate.table(root / 'fakeddit_sample.tsv', rows)
        manifest['samples_sha256'] = io.sha256(root / 'fakeddit_sample.tsv')
        io.write_json(root / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'text'):
            data.fakeddit(root, 'dev', True)
        with self.assertRaisesRegex(ValueError, 'Smoke'):
            data.fakeddit(root, 'test', True)

    def test_metrics_include_missing_classes_and_sample_std(self):
        result = evaluate.metric([0, 0], [0, 0], list(range(3)))
        self.assertEqual(result['accuracy'], 1)
        self.assertAlmostEqual(result['macro_f1'], 1 / 3)
        rows = [{'task': 'x', 'method': 'm', 'samples': 2, 'accuracy': v} for v in [0.0, 1.0]]
        self.assertAlmostEqual(evaluate.summarize(rows)[0]['accuracy_std'], np.sqrt(0.5))

    def test_prepare_commands_do_not_touch_data_or_model(self):
        from vlm_project.cli import main
        with patch('vlm_project.cli.locate_data', side_effect=AssertionError('dataset access')), contextlib.redirect_stdout(stdio.StringIO()):
            main(['benchmark', '--dataset', 'dtd', '--prepare-only'])
            main(['fakeddit', '--split', 'dev', '--prepare-only'])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(stdio.StringIO()):
            main(['fakeddit', '--split', 'test', '--smoke', '--prepare-only'])

    def test_fakeddit_test_guard_before_dataset_and_model(self):
        with patch.object(data, 'fakeddit', side_effect=AssertionError('must not read test')):
            with self.assertRaisesRegex(ValueError, 'freeze'):
                fakeddit.run(self.settings, 'test', 'irrelevant')

    def test_full_benchmark_outputs_shared_image_encoding_and_tamper(self):
        root, rows = self.make_benchmark()
        names = [f'c{k}' for k in range(47)]
        trial = [{'method': 'clip', 'seed': None, 'prompts': {n: [n] for n in names}}, {'method': 'dclip', 'seed': None, 'prompts': {n: [n, 'detail'] for n in names}}]
        trial += [{'method': 'waffle', 'seed': seed, 'prompts': {n: [n, str(seed)] for n in names}} for seed in range(7)]
        source = self.root / 'source/third_party/WaffleCLIP/descriptors'
        source.mkdir(parents=True)
        io.write_json(source / 'descriptors_dtd.json', {n: ['detail'] for n in names})
        FakeBackend.calls = 0
        with patch.object(benchmark, 'ROOT', self.root / 'source'), patch.object(benchmark, 'verify_sources', return_value={}), patch.object(data, 'benchmark', return_value=(names, rows, {'split': 'test-partition-1'})), patch.object(benchmark, 'prompt_trials', return_value=trial), patch.object(benchmark, 'author_api', return_value=None), patch.object(benchmark, 'encode_cohort', side_effect=fake_encode), contextlib.redirect_stdout(stdio.StringIO()):
            output = benchmark.run(self.settings, 'dtd', root, backend_factory=FakeBackend)
        run = protocol.completed_run(output)
        self.assertEqual(FakeBackend.calls, 1)
        self.assertEqual(len(run['summary']), 4)
        self.assertEqual({r['method']: r['seeds'] for r in run['summary']}, {'clip': 1, 'dclip': 1, 'waffle': 7, 'waffle_unit_ablation': 7})
        (output / 'predictions.tsv').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'Artifact'):
            protocol.completed_run(output)

    def test_dev_freeze_test_cycle_and_stale_lock(self):
        rows = [{'id': str(k), 'image': 'synthetic', 'text': f'title{k}', '2way': str(int(k % 6 == 0)), '6way': str(k % 6), 'image_sha256': f'image{k}', 'text_sha256': f'text{k}'} for k in range(3000)]
        spec = fakeddit.profile(self.settings)
        cohort = {'cohort_id': spec['development_source']['dev_cohort_id'], 'split': 'dev'}

        def trials(api, names, descriptor, settings):
            return [{'method': 'dclip', 'seed': None, 'prompts': {n: [n] for n in names}}] + [{'method': 'waffle', 'seed': seed, 'prompts': {n: [n, str(seed)] for n in names}} for seed in range(7)]

        def fast_majority(features, means, seed):
            vote = (features @ means.T).argmax(1)
            return (vote, {'votes': np.stack([vote] * 3, axis=1), 'batch_ids': np.arange(len(vote)) // 75})
        with patch.object(fakeddit, 'verify_sources', return_value={}), patch.object(fakeddit, 'author_api', return_value=None), patch.object(fakeddit, 'prompt_trials', side_effect=trials), patch.object(data, 'fakeddit', return_value=(rows, cohort)), patch.object(fakeddit, 'encode_cohort', side_effect=fake_encode), patch.object(fakeddit, 'majority', side_effect=fast_majority), contextlib.redirect_stdout(stdio.StringIO()):
            output = fakeddit.run(self.settings, 'dev', 'synthetic', backend_factory=FakeBackend)
            lock = protocol.freeze(self.settings, output)
        self.assertEqual(len(protocol.completed_run(output)['summary']), 14)
        self.assertEqual(protocol.validate_lock(self.settings), lock)
        with self.assertRaises(FileExistsError):
            protocol.freeze(self.settings, output)
        with self.assertRaisesRegex(ValueError, 'trùng dev'):
            protocol.check_test_cohort(lock, [rows[0]])
        test_row = {**rows[0], 'id': 'new', 'image_sha256': 'newimage', 'text_sha256': 'newtext'}
        protocol.check_test_cohort(lock, [test_row])
        changed = {**self.settings, 'batch_size': 16}
        with self.assertRaisesRegex(ValueError, 'khác dev'):
            protocol.validate_lock(changed)
        smoke = io.read_json(output / 'run.json')
        smoke['mode'] = 'smoke'
        io.write_json(output / 'run.json', smoke)
        io.write_json(output / 'completed.json', {'status': 'complete', 'run_sha256': io.sha256(output / 'run.json')})
        with self.assertRaisesRegex(ValueError, '3000'):
            protocol.freeze(self.settings, output)

    def test_package_manifest_no_legacy_or_data_and_gitless_cli(self):
        output = self.root / 'package'
        receipt = packaging.package(output)
        with zipfile.ZipFile(receipt['zip']) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read('source_bundle_manifest.json'))
            self.assertFalse(any((n.split('/')[0] in ['data', 'dataraw', 'data_process', 'archive', 'backups', 'experiments', 'third_party', 'venv'] for n in names)))
            for name, expected in manifest['files'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
            extracted = self.root / 'extracted'
            archive.extractall(extracted)
        result = subprocess.run([sys.executable, '-B', 'run.py', 'benchmark', '--dataset', 'dtd', '--prepare-only'], cwd=extracted, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'plan_only')
        notebook = json.loads((output / 'Final_MetaCLIP2_Kaggle.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<notebook>', 'exec')

    def test_kaggle_bootstrap_handles_zip_and_extracted_input(self):
        receipt = packaging.package(self.root / 'bundle')
        boot = ''.join(packaging.notebook()['cells'][2]['source'])
        for mode in ['zip', 'extracted']:
            input_root = self.root / mode / 'input'
            mount = input_root / 'my-code'
            mount.mkdir(parents=True)
            working = self.root / mode / 'working'
            working.mkdir()
            if mode == 'zip':
                import shutil
                shutil.copy2(receipt['zip'], mount / 'final_model_source.zip')
            else:
                with zipfile.ZipFile(receipt['zip']) as archive:
                    archive.extractall(mount)
            portable = boot.replace('"/kaggle/input"', repr(str(input_root)))
            portable = portable.replace('"/kaggle/working"', repr(str(working)))
            namespace = {'TEST_LOCK_INPUT': ''}
            previous = list(sys.path)
            try:
                with patch.object(subprocess, 'run') as installs, contextlib.redirect_stdout(stdio.StringIO()):
                    exec(compile(portable, '<bootstrap>', 'exec'), namespace)
                self.assertEqual(installs.call_count, 2)
                project = namespace['PROJECT']
                self.assertTrue((project / 'run.py').is_file())
                manifest = io.read_json(project / 'source_bundle_manifest.json')
                for name, expected in manifest['files'].items():
                    self.assertEqual(io.sha256(project / name), expected)
                (project / 'run.py').write_text('modified locally')
                with self.assertRaisesRegex(ValueError, 'không ghi đè'), patch.object(subprocess, 'run'):
                    exec(compile(portable, '<bootstrap>', 'exec'), namespace)
            finally:
                sys.path[:] = previous

    def launcher_plan(self, limit=None):
        import yaml
        settings = yaml.safe_load((ROOT / 'configs/kaggle_launcher.yaml').read_text())
        original = kaggle.path
        with patch.object(kaggle, 'path', side_effect=lambda value: self.root / 'launches' if str(value) == 'results/_system/launches/final' else original(value)):
            return kaggle.prepare(settings, 'benchmark_dtd', limit)

    def test_old_style_launcher_prepares_smoke_without_data_or_remote_access(self):
        with patch.object(kaggle, 'cli', side_effect=AssertionError('network not allowed')):
            plan = self.launcher_plan(5)
        self.assertEqual(plan['samples'], 5)
        self.assertEqual(plan['data_dataset'], 'nguyenhung3105/vlm-public-benchmarks-eval-v1')
        self.assertTrue(plan['kernel'].endswith('-smoke'))
        root = Path(plan['root'])
        self.assertEqual({p.name for p in (root / 'code').iterdir()}, {'dataset-metadata.json', 'final_model_source.zip'})
        meta = io.read_json(root / 'kernel/kernel-metadata.json')
        self.assertEqual(meta['enable_gpu'], 'true'); self.assertEqual(meta['is_private'], 'true')
        document = io.read_json(root / 'kernel/run.ipynb')
        code = '\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type'] == 'code')
        self.assertNotIn('output = run(settings, name, root, smoke=False)', code)
        self.assertIn('output = smoke_output', code)
        for c in document['cells']:
            if c['cell_type'] == 'code':compile(''.join(c['source']), '<kaggle>', 'exec')

    def test_launcher_full_submit_reuses_code_and_never_uploads_images(self):
        plan = self.launcher_plan()
        self.assertEqual(plan['samples'], 1880)
        calls = []
        def fake_cli(arguments, check=True):
            calls.append(arguments)
            if arguments[:2] == ['kernels', 'status']:
                return subprocess.CompletedProcess(arguments, 1, '', '404 not found')
            if arguments[:2] == ['datasets', 'status']:
                return subprocess.CompletedProcess(arguments, 0, '{"status":"ready"}', '')
            return subprocess.CompletedProcess(arguments, 0, 'pushed', '')
        with patch.object(kaggle, 'cli', side_effect=fake_cli),contextlib.redirect_stdout(stdio.StringIO()):
            result = kaggle.submit(plan)
        self.assertEqual(result['status'], 'submitted')
        self.assertEqual(len([c for c in calls if c[:2] == ['kernels','push']]), 1)
        self.assertFalse(any(c[:2] in [['datasets','version'],['datasets','create']] for c in calls))

    def test_launcher_refuses_duplicate_running_job(self):
        plan = self.launcher_plan()
        active = subprocess.CompletedProcess([], 0, 'Kernel has status "running"', '')
        with patch.object(kaggle, 'cli', return_value=active) as mock:
            with self.assertRaisesRegex(RuntimeError, 'đang chạy'):kaggle.submit(plan)
        self.assertEqual(mock.call_count, 1)

    def test_launcher_new_kernel_permission_denied_checks_owned_list_then_pushes(self):
        plan = self.launcher_plan()
        calls = []
        def fake_cli(arguments, check=True):
            calls.append(arguments)
            if arguments[:2] == ['kernels', 'status']:
                return subprocess.CompletedProcess(arguments, 1, '',
                    "Cannot access kernel 'nguyenhung3105/vlm-final-dtd-metaclip2' (Permission 'kernels.get' was denied).")
            if arguments[:2] == ['kernels', 'list']:
                return subprocess.CompletedProcess(arguments, 0,
                    '[{"ref":"nguyenhung3105/my-old-notebook"}]', '')
            if arguments[:2] == ['datasets', 'status']:
                return subprocess.CompletedProcess(arguments, 0, '{"status":"ready"}', '')
            return subprocess.CompletedProcess(arguments, 0, 'pushed', '')
        with patch.object(kaggle, 'cli', side_effect=fake_cli), contextlib.redirect_stdout(stdio.StringIO()):
            kaggle.submit(plan)
        self.assertEqual(len([c for c in calls if c[:2] == ['kernels', 'list']]), 1)
        self.assertEqual(len([c for c in calls if c[:2] == ['kernels', 'push']]), 1)

    def test_launcher_denied_existing_kernel_never_pushes(self):
        plan = self.launcher_plan()
        for message in ["Permission 'kernels.get' was denied", "403 Permission 'kernels.get' was denied; resource not found"]:
            denied = subprocess.CompletedProcess([], 1, '', message)
            listing = subprocess.CompletedProcess([], 0, json.dumps([{'ref': plan['kernel']}]), '')
            with patch.object(kaggle, 'cli', side_effect=[denied, listing]) as mock:
                with self.assertRaisesRegex(RuntimeError, 'kernels.get'):
                    kaggle.submit(plan)
            self.assertEqual(mock.call_count, 2)

    def test_launcher_denied_list_or_wrong_account_never_uploads(self):
        for listing in ['authentication failed', json.dumps([{'ref': 'another-account/notebook'}])]:
            plan = self.launcher_plan()
            denied = subprocess.CompletedProcess([], 1, '', "Permission 'kernels.get' was denied")
            def fake_cli(arguments, check=True):
                if arguments[:2] == ['kernels', 'status']:return denied
                if listing == 'authentication failed':raise RuntimeError(listing)
                return subprocess.CompletedProcess(arguments, 0, listing, '')
            with patch.object(kaggle, 'cli', side_effect=fake_cli) as mock:
                with self.assertRaises(RuntimeError):kaggle.submit(plan)
            self.assertEqual(mock.call_count, 2)

    def test_owned_list_paginates_before_deciding_kernel_is_missing(self):
        reference = 'nguyenhung3105/existing-private-notebook'
        first = [{'ref':f'nguyenhung3105/old-{i}'} for i in range(100)]
        second = [{'ref':reference}]
        responses = [subprocess.CompletedProcess([],0,json.dumps(rows),'') for rows in [first,second]]
        with patch.object(kaggle,'cli',side_effect=responses) as mock:
            self.assertTrue(kaggle.owned_reference_exists('kernels',reference))
        self.assertEqual(mock.call_count,2)
        self.assertEqual(mock.call_args_list[1].args[0][-2:],['--page','2'])

    def test_new_code_dataset_permission_denied_creates_only_after_owned_list(self):
        plan = self.launcher_plan(); calls=[]; lookups=0
        def fake_cli(arguments,check=True):
            nonlocal lookups
            calls.append(arguments)
            if arguments[:2]==['kernels','status']:
                return subprocess.CompletedProcess(arguments,1,'','404 not found')
            if arguments[:3]==['datasets','status',plan['code_dataset']]:
                lookups+=1
                if lookups==1:return subprocess.CompletedProcess(arguments,1,'','403 Forbidden')
            if arguments[:2]==['datasets','list']:
                return subprocess.CompletedProcess(arguments,0,'[]','')
            return subprocess.CompletedProcess(arguments,0,'{"status":"ready"}','')
        with patch.object(kaggle,'cli',side_effect=fake_cli),contextlib.redirect_stdout(stdio.StringIO()):
            kaggle.submit(plan)
        self.assertEqual(len([c for c in calls if c[:2]==['datasets','create']]),1)
        self.assertFalse(any(c[:2]==['datasets','version'] for c in calls))

    def test_unknown_or_malformed_owned_list_stops_before_submit(self):
        for output in ['not JSON', '{"message":"permission denied"}', '[{"title":"unknown"}]']:
            with patch.object(kaggle,'cli',return_value=subprocess.CompletedProcess([],0,output,'')):
                with self.assertRaises(ValueError):
                    kaggle.owned_reference_exists('kernels','nguyenhung3105/new-notebook')

    def test_launcher_creates_only_code_dataset_when_missing(self):
        plan = self.launcher_plan()
        calls=[];code_lookups=0
        def fake_cli(arguments, check=True):
            nonlocal code_lookups
            calls.append(arguments)
            if arguments[:2] == ['kernels','status']:
                return subprocess.CompletedProcess(arguments,1,'','404 not found')
            if arguments[:3] == ['datasets','status',plan['code_dataset']]:
                code_lookups+=1
                if code_lookups==1:return subprocess.CompletedProcess(arguments,1,'','404 not found')
            return subprocess.CompletedProcess(arguments,0,'{"status":"ready"}','')
        with patch.object(kaggle,'cli',side_effect=fake_cli),contextlib.redirect_stdout(stdio.StringIO()):
            kaggle.submit(plan)
        created=[c for c in calls if c[:2]==['datasets','create']]
        self.assertEqual(len(created),1)
        self.assertEqual(Path(created[0][3]).resolve(),(Path(plan['root'])/'code').resolve())

    def test_launcher_wait_downloads_and_verifies_metrics(self):
        plan = self.launcher_plan()
        source = self.root / 'remote_run'
        source.mkdir()
        records = [{'task':'dtd','method':'clip','seed':None,'samples':2,
                    'accuracy':1.,'macro_f1':1.,'balanced_accuracy':1.}]
        predictions = [{'task':'dtd','method':'clip','seed':None,'id':str(k),'truth':k,'predicted':k} for k in range(2)]
        manifest = {'model':'metaclip2_vith14','dataset':'dtd','mode':'full','samples':2}
        with contextlib.redirect_stdout(stdio.StringIO()):
            evaluate.publish(source,records,predictions,{},manifest)
        remote = self.root / 'metrics_test_synthetic.zip'
        with zipfile.ZipFile(remote,'w') as archive:
            for file in source.iterdir():archive.write(file,'run/'+file.name)
        def fake_cli(arguments,check=True):
            if arguments[:2]==['kernels','output']:
                import shutil
                target = Path(arguments[arguments.index('-p')+1])
                shutil.copy2(remote,target/remote.name)
            return subprocess.CompletedProcess(arguments,0,'complete','')
        original = kaggle.path
        with patch.object(kaggle,'cli',side_effect=fake_cli),contextlib.redirect_stdout(stdio.StringIO()),\
             patch.object(kaggle,'path',side_effect=lambda value:self.root/'downloads' if str(value)=='results/final/kaggle' else original(value)):
            collected = kaggle.wait_and_collect(plan)
        self.assertEqual(collected['status'],'collected')
        self.assertEqual(protocol.completed_run(Path(collected['output'])/'run')['samples'],2)
if __name__ == '__main__':
    unittest.main()
