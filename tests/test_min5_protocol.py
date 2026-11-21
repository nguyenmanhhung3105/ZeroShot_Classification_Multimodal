"""Protocol min5 dùng artifact tổng hợp; không đọc train/dev/test thật."""
import contextlib
import copy
import io as stdio
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_project import io, fewshot_support_thresholds as thresholds
from vlm_project import min5_protocol as protocol
import test_fewshot_support_thresholds as fixtures


class Min5ProtocolTests(unittest.TestCase):
    def reseal_run(self, directory):
        run = io.read_json(directory / 'run.json')
        for name in run['output_sha256']:
            run['output_sha256'][name] = io.sha256(directory / name)
        io.write_json(directory / 'run.json', run)
        io.write_json(directory / 'completed.json', {'status': 'complete', 'run_sha256': io.sha256(directory / 'run.json')})

    def fixture(self, root):
        helper = fixtures.ThresholdTests()
        source, ids, six, scores = helper.source(root)
        train = [{'id': f't{i}', '6way': int(i % 6), 'image_sha256': io.digest(['train', i]),
                  'text_sha256': io.digest(['train text', i])} for i in range(3000)]
        dev = [{'id': str(id_), '6way': int(y), 'image_sha256': io.digest(['dev', i]),
                'text_sha256': io.digest(['dev text', i])} for i, (id_, y) in enumerate(zip(ids, six))]
        io.write_json(source / 'train_cohort.json', train)
        io.write_json(source / 'cohort.json', dev)
        log = []
        for policy in thresholds.POLICIES:
            for seed in range(5):
                quota = [64] * 6 if policy == 'legacy_6way' else [192, 38, 38, 38, 38, 38]
                if policy != 'legacy_6way':
                    for offset in range(2): quota[1+(seed+offset) % 5] += 1
                rows = [r for y, count in enumerate(quota) for r in [x for x in train if x['6way'] == y][:count]]
                log.append({'policy': policy, 'seed': seed, 'ids': [r['id'] for r in rows],
                            'labels_6way': [r['6way'] for r in rows], 'quotas_6way': quota})
        io.write_json(source / 'support.json', log)
        helper.seal(source)
        run = io.read_json(source / 'run.json')
        run.update(support_sha256=io.digest(log), feature='synthetic feature',
                   specification_sha256='a'*64, model_runtime={'architecture': 'synthetic'})
        run['output_sha256'].update({n: io.sha256(source / n) for n in ['support.json', 'train_cohort.json']})
        io.write_json(source / 'run.json', run)
        io.write_json(source / 'completed.json', {'status': 'complete', 'run_sha256': io.sha256(source / 'run.json')})
        with contextlib.redirect_stdout(stdio.StringIO()):
            review = thresholds.analyze(source, root / 'review')
        settings = root / 'settings.yaml'
        settings.write_text('model: synthetic\nbatch_size: 8\n')
        return source, review, settings, train, dev

    def test_prepare_seal_validate_without_test_reads_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, review, settings, train, dev = self.fixture(root)
            draft = protocol.prepare(source, review, root / 'draft.json', settings)
            self.assertEqual(draft['payload']['state'], 'draft')
            self.assertEqual(len(draft['payload']['candidates']), 20)
            self.assertEqual(len(draft['payload']['excluded_train']), 3000)
            self.assertEqual(len(draft['payload']['excluded_dev']), 3000)
            with self.assertRaisesRegex(ValueError, 'draft'):
                protocol.validate(root / 'draft.json')
            with self.assertRaises(FileExistsError):
                protocol.prepare(source, review, root / 'draft.json', settings)
            protocol.seal(root / 'draft.json', 'b'*64, root / 'lock.json')
            lock = protocol.validate(root / 'lock.json')
            self.assertEqual(lock['payload']['test_manifest_sha256'], 'b'*64)
            with self.assertRaises(FileExistsError):
                protocol.seal(root / 'draft.json', 'b'*64, root / 'lock.json')
            changed = copy.deepcopy(lock)
            changed['payload']['candidates'][0]['seed'] = 99
            io.write_json(root / 'tampered.json', changed)
            with self.assertRaises(ValueError): protocol.validate(root / 'tampered.json')
            settings.write_text('model: synthetic\nbatch_size: 16\n')
            with self.assertRaisesRegex(ValueError, 'Code|cấu hình'):
                protocol.validate(root / 'lock.json')

    def test_guards_test_metadata_overlap_and_code_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, review, settings, train, dev = self.fixture(root)
            protocol.prepare(source, review, root / 'draft.json', settings)
            protocol.seal(root / 'draft.json', 'b'*64, root / 'lock.json')
            lock = protocol.validate(root / 'lock.json')
            rows = [{'id': f'new{i}', 'image_sha256': io.digest(['new', i]),
                     'text_sha256': io.digest(['new text', i])} for i in range(3000)]
            meta = {'split': 'test', 'source_split': 'test_public', 'manifest_sha256': 'b'*64}
            protocol.check_test_cohort(lock, rows, meta)
            for key in ['id', 'image_sha256', 'text_sha256']:
                for excluded in [train, dev]:
                    bad = copy.deepcopy(rows); bad[0][key] = excluded[0][key]
                    with self.assertRaisesRegex(ValueError, 'overlap'):
                        protocol.check_test_cohort(lock, bad, meta)
            for update in [{'split': 'dev'}, {'manifest_sha256': 'c'*64}, {'source_split': 'validate'}]:
                with self.assertRaises(ValueError): protocol.check_test_cohort(lock, rows, {**meta, **update})
            with self.assertRaises(ValueError): protocol.check_test_cohort(lock, rows[:-1], meta)
            bad = copy.deepcopy(rows); bad[1]['id'] = bad[0]['id']
            with self.assertRaises(ValueError): protocol.check_test_cohort(lock, bad, meta)
            with patch.object(protocol, 'fingerprint', return_value={}):
                with self.assertRaises(ValueError): protocol.validate(root / 'lock.json')

    def test_reject_invalid_test_hash_and_threshold_source_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, review, settings, _, _ = self.fixture(root)
            protocol.prepare(source, review, root / 'draft.json', settings)
            for value in ['', 'v2', 'a'*63]:
                with self.assertRaises(ValueError): protocol.seal(root / 'draft.json', value, root / 'lock.json')
            run = io.read_json(source / 'run.json'); run['feature'] = 'changed'
            io.write_json(source / 'run.json', run)
            io.write_json(source / 'completed.json', {'status': 'complete', 'run_sha256': io.sha256(source / 'run.json')})
            with self.assertRaises(ValueError): protocol.prepare(source, review, root / 'other.json', settings)

    def test_support_must_be_unique_train_only_correct_quota_and_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _, _, train, _ = self.fixture(root)
            original = io.read_json(source / 'support.json')
            for case in ['duplicate_entry', 'missing_entry', 'duplicate_id', 'outside_train', 'labels', 'quota', 'seed']:
                with self.subTest(case=case):
                    log = copy.deepcopy(original)
                    if case == 'duplicate_entry': log.append(log[0])
                    if case == 'missing_entry': log.pop()
                    if case == 'duplicate_id': log[0]['ids'][1] = log[0]['ids'][0]
                    if case == 'outside_train': log[0]['ids'][0] = 'outside'
                    if case == 'labels': log[0]['labels_6way'][0] = 5
                    if case == 'quota': log[0]['quotas_6way'][0] = 63
                    if case == 'seed': log[0]['seed'] = 9
                    with self.assertRaises(ValueError): protocol.validate_support(log, train)

    def test_threshold_tampering_is_detected_even_if_review_is_rehashed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, review, settings, _, _ = self.fixture(root)
            p = review / 'thresholds.tsv'
            import csv
            with p.open() as stream: rows = list(csv.DictReader(stream, delimiter='\t'))
            from vlm_project.evaluate import table
            rows[0]['threshold'] = '.999'
            table(p, rows)
            self.reseal_run(review)
            with self.assertRaisesRegex(ValueError, 'selection'):
                protocol.prepare(source, review, root / 'draft.json', settings)
            self.assertFalse((root / 'draft.json').exists())

    def test_cli_draft_is_not_a_test_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, review, settings, _, _ = self.fixture(root)
            script = ROOT / 'scripts/prepare_min5_protocol.py'
            run = subprocess.run([sys.executable, '-B', str(script), 'prepare', '--source-run', str(source),
                                  '--threshold-run', str(review), '--config', str(settings), '--output', str(root / 'draft.json')],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('Protocol draft:', run.stdout)
            check = [sys.executable, '-B', str(script), 'validate', '--protocol', str(root / 'draft.json')]
            self.assertNotEqual(subprocess.run(check, capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run([*check, '--allow-draft'], capture_output=True).returncode, 0)

    def test_identity_checksum_and_bad_state_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, review, settings, train, _ = self.fixture(root)
            for rows in [train[:-1], [{**train[0], 'image_sha256': 'bad'}, *train[1:]],
                         [{**train[0], 'id': ''}, *train[1:]]]:
                with self.assertRaises(ValueError): protocol.identities(rows)
            protocol.prepare(source, review, root / 'draft.json', settings)
            for update in [{'state': 'unknown'}, {'state': 'sealed_task1', 'test_manifest_sha256': None}]:
                lock = io.read_json(root / 'draft.json'); lock['payload'].update(update)
                lock['sha256'] = io.digest(lock['payload']); io.write_json(root / 'bad.json', lock)
                with self.assertRaises(ValueError): protocol.validate(root / 'bad.json')
            (source / 'cohort.json').write_text('[]')
            with self.assertRaises(ValueError): protocol.prepare(source, review, root / 'other.json', settings)

    def test_test_guard_rejects_invalid_schema_even_with_consistent_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, review, settings, _, _ = self.fixture(root)
            protocol.prepare(source, review, root / 'draft.json', settings)
            original = protocol.seal(root / 'draft.json', 'b'*64, root / 'lock.json')
            rows = [{'id': f'new{i}', 'image_sha256': io.digest(['new', i]),
                     'text_sha256': io.digest(['new text', i])} for i in range(3000)]
            metadata = {'split': 'test', 'source_split': 'test_public', 'manifest_sha256': 'b'*64}
            for update in [{'version': 99}, {'kind': 'legacy'}, {'test_opened': True},
                           {'test_manifest_sha256': None}]:
                with self.subTest(update=update):
                    lock = copy.deepcopy(original)
                    lock['payload'].update(update); lock['sha256'] = io.digest(lock['payload'])
                    meta = {**metadata, 'manifest_sha256': update.get('test_manifest_sha256', 'b'*64)}
                    with self.assertRaises(ValueError): protocol.check_test_cohort(lock, rows, meta)

    def test_relocated_same_config_works_for_both_validation_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, review, settings, _, _ = self.fixture(root)
            protocol.prepare(source, review, root / 'draft.json', settings)
            protocol.seal(root / 'draft.json', 'b'*64, root / 'lock.json')
            relocated = root / 'relocated.yaml'; relocated.write_bytes(settings.read_bytes())
            settings.unlink()
            lock = protocol.validate(root / 'lock.json', settings_file=relocated)
            rows = [{'id': f'new{i}', 'image_sha256': io.digest(['new', i]),
                     'text_sha256': io.digest(['new text', i])} for i in range(3000)]
            metadata = {'split': 'test', 'source_split': 'test_public', 'manifest_sha256': 'b'*64}
            protocol.check_test_cohort(lock, rows, metadata, settings_file=relocated)


if __name__ == '__main__': unittest.main()
