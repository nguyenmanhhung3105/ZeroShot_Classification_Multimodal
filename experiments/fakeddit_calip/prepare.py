"""Freeze prompts and existing baseline cohort; never open dataset/images."""
import csv
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from experiments.waffle_lab.run import digest, file_hash, write_json
from experiments.waffle_lab.core import attach_semantics
from research.upstream import author_api, generate_prompts
from prompts.fakeddit_prompts import get_label_names

ASSET = 'configs/prompts/fakeddit_calip_reference.json'
SEMANTIC = 'configs/prompts/fakeddit_calip_semantic.json'
SOURCES = {
    '2way': 'results/baseline/fakeddit/2way/samples_3000/metaclip2_vith14/2way_16ead71cd477/run_artifacts.zip',
    '6way': 'results/baseline/fakeddit/6way/samples_3000/metaclip2_vith14/6way_c66fec810255/run_artifacts.zip',
}


def verify_prompt_sources():
    sources = json.loads((ROOT/'configs/research/sources.json').read_text())
    for name, spec in sources.items():
        repo = ROOT/'third_party'/name
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        lines = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).splitlines()
        # Finder metadata is not imported or used by prompt generation.
        changes = [line for line in lines if line != '?? .DS_Store']
        if commit != spec['commit'] or changes:
            raise ValueError('Prompt source changed: '+name)
    return sources


def load_reference(path):
    ref = json.loads(Path(path).read_text())
    if ref['sha256'] != digest(ref['payload']):
        raise ValueError('Fakeddit reference fingerprint mismatch')
    return ref


def prepare():
    path = ROOT/ASSET
    if path.exists():
        return load_reference(path)
    spec = json.loads((ROOT/SEMANTIC).read_text())
    api = author_api()
    payload = {'model': 'metaclip2_vith14', 'prompt_seed': 0, 'grouping_seeds': [0, 1, 2],
               'query_size': 75, 'semantic_sha256': file_hash(ROOT/SEMANTIC),
               'sources': verify_prompt_sources(), 'tasks': {}, 'label_fit': False}
    for task, source in SOURCES.items():
        with zipfile.ZipFile(ROOT/source) as z:
            def read(suffix):
                names = [n for n in z.namelist() if n.endswith('/'+suffix)]
                if len(names) != 1: raise ValueError('Ambiguous baseline artifact')
                return z.read(names[0]).decode()
            snap = json.loads(read('evaluation_snapshot.json'))
            manifest = json.loads(read('manifest.json'))
            predictions = list(csv.DictReader(io.StringIO(read('raw_predictions.tsv')), delimiter='\t'))
        if (manifest['model'] != payload['model'] or manifest['prompt_aggregation'] != 'mean_sim'
                or manifest['relation_enabled'] or len(predictions) != 3000):
            raise ValueError('Unexpected baseline protocol')
        names = {str(k): v for k, v in get_label_names(task).items()}
        descriptions = spec[task]
        with tempfile.TemporaryDirectory() as tmp:
            desc = Path(tmp)/'descriptors.json'
            write_json(desc, {name: descriptions[k] for k, name in names.items()})
            author, _ = generate_prompts(api, 'waffle', names.values(), desc, 15, 0)
        clean, noisy = attach_semantics(names, descriptions, author, 15)
        banks = {'legacy': manifest['rendered_prompts'], 'semantic': clean, 'semantic_waffle': noisy}
        for bank in banks.values():
            for prompts in bank.values(): api.clip.tokenize(prompts, truncate=False)
        samples = snap['cohort']['samples']
        if len(samples) != len(predictions): raise ValueError('Baseline sample count mismatch')
        rows = []
        for row, pred in zip(samples, predictions):
            if row['id'] != pred['id'] or row['label'] != pred['y_true']:
                raise ValueError('Baseline row alignment mismatch')
            rows.append({**row, 'baseline_pred': pred['y_pred'],
                         'baseline_image_prob': [float(pred['score_image_'+k]) for k in names]})
        payload['tasks'][task] = {'labels': list(names), 'banks': banks, 'rows': rows,
                                 'baseline_zip_sha256': file_hash(ROOT/source),
                                 'baseline_source': source, 'sample_set_id': snap['sample_set_id'],
                                 'image_weight': manifest['image_weight'], 'logit_scale': manifest['logit_scale'],
                                 'source_split': snap['effective_dataset'].get('source_split')}
    a, b = [payload['tasks'][t]['rows'] for t in ['2way', '6way']]
    if ([r['id'] for r in a] != [r['id'] for r in b] or
            any(x['label'] != ('1' if y['label'] == '0' else '0') for x, y in zip(a, b))):
        raise ValueError('2way/6way cohort or taxonomy relation mismatch')
    ref = {'payload': payload, 'sha256': digest(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(ref, f, ensure_ascii=False, indent=2)
    print('[PREPARED]', path, 'no dataset/image/model access')
    return ref


if __name__ == '__main__': prepare()
