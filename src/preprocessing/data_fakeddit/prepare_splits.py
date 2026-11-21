"""Build three disjoint Fakeddit cohorts, preserving official source splits.

Streams annotations, samples by six-way source proportions, downloads only
needed candidates, resumes interrupted builds, and publishes only after audit.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from preprocessing.data_fakeddit.helpers import digest, image_ok, quotas, text_value, write_tsv

SOURCES = {'train': 'multimodal_train.tsv', 'dev': 'multimodal_validate.tsv', 'test': 'multimodal_test_public.tsv'}
COLUMNS = ['id', 'clean_title', 'image_url', 'hasImage', '2_way_label', '6_way_label']


def network_available():
    """Distinguish an expired source hostname from loss of network access."""
    try:
        response=subprocess.run(['curl','--head','--silent','--max-time','5','https://i.imgur.com/'],
                                capture_output=True,timeout=7)
        return response.returncode == 0
    except (OSError,subprocess.TimeoutExpired):return False


def fetch_image(row, cache, timeout):
    """curl bounds DNS, connect and the entire transfer; publish valid bytes only."""
    if not shutil.which('curl'):raise RuntimeError('curl is required for bounded image downloads')
    with tempfile.NamedTemporaryFile(prefix='.download-',suffix='.part',dir=cache,delete=False) as stream:
        temporary=Path(stream.name)
    try:
        response=subprocess.run(['curl','--silent','--show-error','--location','--fail',
                                 '--max-redirs','5','--connect-timeout',str(min(timeout,4)),
                                 '--max-time',str(timeout),'--max-filesize',str(20*1024*1024),
                                 '--user-agent','Mozilla/5.0','--output',str(temporary),
                                 '--write-out','%{http_code}\n%{url_effective}',
                                 '--url',row['image_url']],capture_output=True,text=True,timeout=timeout+2)
        if response.returncode:
            if response.returncode==6:return None,'dns_unavailable'
            if response.returncode==22:return None,'http_'+response.stdout.splitlines()[0]
            return None,'download_timeout' if response.returncode==28 else 'connection_error'
        if temporary.stat().st_size>20*1024*1024:return None,'image_too_large'
        effective=response.stdout.splitlines()[-1]
        if Path(urlsplit(effective).path).name.lower() in ['removed.png','removed.jpg']:
            return None,'removed_image'
        try:image_signature(temporary,min_side=1)
        except (OSError,ValueError,Image.DecompressionBombError):return None,'invalid_download_image'
        target=cache/(row['id']+'.jpg');temporary.replace(target)
        return target,None
    except subprocess.TimeoutExpired:return None,'download_timeout'
    finally:temporary.unlink(missing_ok=True)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def canonical_url(value):
    p = urlsplit(value.strip())
    if p.scheme.lower() not in ['http', 'https'] or not p.netloc:
        raise ValueError('Invalid image URL')
    return urlunsplit(('https', p.netloc.lower(), p.path, p.query, ''))


def read_metadata(path, min_text_length, rejected):
    rows = []; seen = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if not set(COLUMNS) <= set(reader.fieldnames or []):
            raise ValueError(f'Missing columns in {path}: {sorted(set(COLUMNS)-set(reader.fieldnames or []))}')
        for number, raw in enumerate(reader, 2):
            key = (raw.get('id') or '').strip()
            a, b = raw.get('2_way_label'), raw.get('6_way_label')
            text = text_value(raw.get('clean_title'))
            if not re.fullmatch(r'[A-Za-z0-9_-]+', key): rejected['invalid_id'] += 1; continue
            if key in seen: rejected['duplicate_id'] += 1; continue
            seen.add(key)
            if a not in ['0', '1'] or b not in list('012345') or (a == '1') != (b == '0'):
                rejected['invalid_or_inconsistent_label'] += 1; continue
            if len(text.replace(' ', '')) <= min_text_length:
                rejected['short_or_empty_title'] += 1; continue
            if (raw.get('hasImage') or '').lower() != 'true':
                rejected['no_image'] += 1; continue
            try: url = canonical_url(raw.get('image_url') or '')
            except ValueError: rejected['invalid_url'] += 1; continue
            rows.append({'id': key, '_key': key, 'clean_title': text, 'image_url': url,
                         'hasImage': 'True', '2_way_label': a, '6_way_label': b,
                         'source_row': number,
                         'text_sha256': hashlib.sha256(text.casefold().encode()).hexdigest()})
    return rows


def image_signature(path, min_side=64):
    with Image.open(path) as header:
        if header.width*header.height > 20_000_000: raise ValueError('image_pixel_limit')
    if not image_ok(path): raise ValueError('broken_or_placeholder_image')
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
        width, height = image.size
        if min(width, height) < min_side: raise ValueError('image_too_small')
        pixels = hashlib.sha256(f'{width}x{height}:'.encode()+image.tobytes()).hexdigest()
        gray = np.asarray(image.resize((9, 8)).convert('L'))
        bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
        dhash = sum(int(bit) << k for k, bit in enumerate(bits))
        # Low-detail images have many hash collisions; use exact pixels for them.
        near = f'{dhash:016x}' if float(gray.std()) >= 5 else None
    return {'image_sha256': digest(path), 'pixel_sha256': pixels, 'dhash': near,
            'width': width, 'height': height}


class DisjointIndex:
    def __init__(self, near_distance=2):
        self.ids = set(); self.urls = set(); self.text = set(); self.bytes = set(); self.pixels = set()
        self.bands = defaultdict(set); self.near_distance = near_distance

    def metadata_reason(self, row):
        for key, values in [('id', self.ids), ('image_url', self.urls), ('text_sha256', self.text)]:
            if row[key] in values: return 'duplicate_'+key
        return None

    def image_reason(self, signature):
        if signature['image_sha256'] in self.bytes: return 'duplicate_image_bytes'
        if signature['pixel_sha256'] in self.pixels: return 'duplicate_image_pixels'
        value = signature.get('dhash')
        if value and self.near_distance >= 0:
            h = int(value, 16); candidates = set()
            for band in range(4): candidates |= self.bands[(band, (h >> (16*band)) & 65535)]
            if any((h ^ other).bit_count() <= self.near_distance for other in candidates):
                return 'near_duplicate_image'
        return None

    def add(self, row):
        self.ids.add(row['id']); self.urls.add(row['image_url']); self.text.add(row['text_sha256'])
        self.bytes.add(row['image_sha256']); self.pixels.add(row['pixel_sha256'])
        if row.get('dhash'):
            h = int(row['dhash'], 16)
            for band in range(4): self.bands[(band, (h >> (16*band)) & 65535)].add(h)


def old_development_exclusions(root):
    """Previously inspected validate examples may not reappear in the new test."""
    ids = set(); images = set(); texts = set()
    ref = root/'configs/prompts/fakeddit_calip_reference.json'
    if ref.is_file():
        spec = json.loads(ref.read_text())['payload']
        for plan in spec['tasks'].values():
            for row in plan['rows']:
                ids.add(row['id']); texts.add(row['text_sha256'])
    # Existing result artifact, not an image/TSV dataset scan.
    import zipfile
    old = root/'results/fakeddit_calip/fakeddit/6way/samples_3000/metaclip2_vith14/6way_a3f7a8a3a561/run_artifacts.zip'
    if old.is_file():
        with zipfile.ZipFile(old) as z:
            for row in json.loads(z.read('cohort.json')): images.add(row['sha256'])
    return {'ids': ids, 'images': images, 'texts': texts}


def materialize(row, root, assets, download, timeout, min_side):
    locations = [assets/f"{row['id']}.jpg"]
    for folder in ['data/processed/pro_fakeddit/images', 'data/raw/fakeddit/images']:
        locations.extend(root/folder/f"{row['id']}{ext}" for ext in ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG'])
    for path in locations:
        if path.is_file():
            try: return path, image_signature(path, min_side), None, False
            except (ValueError, OSError, Image.DecompressionBombError): continue
    if not download: return None, None, 'missing_local_image', False
    path, error = fetch_image(row, assets, timeout)
    if error: return None, None, error, True
    try: return path, image_signature(path, min_side), None, True
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        return None, None, str(exc) if isinstance(exc, ValueError) else 'invalid_image', True


def split_config(base, root_relative, logical, size):
    config = deepcopy(base)
    for data in config['datasets'].values(): data['enabled'] = False
    data = config['datasets']['fakeddit']
    folder = root_relative/logical
    data.update(enabled=True, task='6way', data_path=str(folder/'fakeddit_sample.tsv'),
                image_dir=str(folder/'images'), id_col='id', text_col='clean_title',
                label_col_2way='2_way_label', label_col_6way='6_way_label', min_text_length=5,
                max_samples=size, source_split={'train':'train', 'dev':'validate', 'test':'test_public'}[logical],
                label_source='6_way_label', cohort_manifest=str(folder/'manifest.json'),
                fakeddit_split_protocol='v2')
    data['image_text_relation'] = {'enabled': False}
    config.setdefault('evaluation', {}).update(split=logical, sample_lock_mode='create_or_verify',
                                               sample_lock_dir='results/_system/sample_locks/fakeddit_splits_v2')
    config.setdefault('output', {}).update(results_dir='auto', raw_predictions_dir='auto', summary_table='auto')
    return config


def verify(output, size=3000, min_side=64, near_distance=2):
    output = Path(output); index = DisjointIndex(near_distance); result = {}
    for logical in ['test', 'dev', 'train']:
        manifest = json.loads((output/logical/'manifest.json').read_text())
        path = output/logical/'fakeddit_sample.tsv'
        if digest(path) != manifest['samples_sha256']: raise ValueError('TSV checksum mismatch')
        with path.open(newline='') as f: rows = list(csv.DictReader(f, delimiter='\t'))
        if (len(rows) != size or len(manifest['samples']) != size or manifest['selected']['samples'] != size
                or manifest['cohort_id'] != object_hash(manifest['samples'])[:20]):
            raise ValueError(f'{logical}: incomplete or inconsistent manifest/cohort')
        support = Counter()
        for row, expected in zip(rows, manifest['samples']):
            if row['id'] != expected['id'] or row['6_way_label'] != str(expected['y_true']): raise ValueError('Manifest row mismatch')
            if (row['2_way_label'] not in ['0','1'] or row['6_way_label'] not in list('012345')
                    or (row['2_way_label'] == '1') != (row['6_way_label'] == '0')): raise ValueError('2way/6way mismatch')
            expected_split={'train':'train','dev':'validate','test':'test_public'}[logical]
            text_hash=hashlib.sha256(text_value(row['clean_title']).casefold().encode()).hexdigest()
            if (row['split'] != logical or row['source_split'] != expected_split or manifest['source_split'] != expected_split
                    or row['source_file'] != manifest['source']['path'] or row['hasImage'].lower() != 'true'
                    or row['text_sha256'] != text_hash or expected['text_sha256'] != text_hash
                    or canonical_url(row['image_url']) != row['image_url']):
                raise ValueError('Source provenance/title metadata mismatch')
            if index.metadata_reason(row): raise ValueError('Cross-split/within-split metadata duplicate')
            relative = Path(row['image_path'])
            if relative.parts[-3:] != (logical,'images',row['id']+'.jpg'):
                raise ValueError('Invalid exported image path')
            # Resolve under the current build/output root, so pre-publication audits work.
            image_path = output/logical/'images'/relative.name
            signature = image_signature(image_path, min_side)
            if signature['image_sha256'] != expected['image_sha256']: raise ValueError('Image checksum mismatch')
            if index.image_reason(signature): raise ValueError('Cross-split/within-split image duplicate')
            index.add({**row, **signature}); support[row['6_way_label']] += 1
        if (dict(support) != manifest['selected']['support'] or set(support) != set('012345')
                or dict(support) != manifest['source']['quota']):
            raise ValueError('Missing class or support mismatch')
        result[logical] = {'samples': len(rows), 'support': dict(sorted(support.items())), 'images_verified': len(rows)}
    return {'status': 'passed', 'splits': result, 'total_samples': 3*size,
            'disjoint': ['id', 'canonical_image_url', 'normalized_title', 'image_bytes', 'image_pixels'],
            'near_duplicate_max_hamming': near_distance}


def prepare(args, root=ROOT):
    root = Path(root).resolve(); output = (root/args.output).resolve()
    if not output.is_relative_to(root/'data/processed'): raise ValueError('Output must be under data/processed')
    if output.exists() and not args.inspect: return verify(output, args.samples, args.min_side, args.near_distance)
    source = root/'data/raw/fakeddit'; pools = {}; source_meta = {}; rejected = {}
    for logical, filename in SOURCES.items():
        path = source/filename
        if not path.is_file(): raise FileNotFoundError(f'Missing official source: {path}; source splits will not be silently mixed')
        errors = Counter(); rows = read_metadata(path, args.min_text_length, errors)
        counts = Counter(r['6_way_label'] for r in rows)
        if set(counts) != set('012345'): raise ValueError(f'{logical}: missing source classes')
        wanted = quotas(counts, args.samples)
        if min(wanted.values()) < 1: raise ValueError('A class rounded to zero; choose a larger cohort')
        pools[logical] = rows
        source_meta[logical] = {'path': str(path.relative_to(root)), 'sha256': digest(path),
                                'valid_metadata': len(rows), 'support': dict(sorted(counts.items())), 'quota': wanted}
        rejected[logical] = errors
        print('[SOURCE]', logical, len(rows), 'quota', wanted, flush=True)
    settings = {'samples': args.samples, 'seed': args.seed, 'min_text_length': args.min_text_length,
                'min_side': args.min_side, 'near_distance': args.near_distance, 'sources': source_meta}
    if args.inspect:
        return {'status':'metadata_only', **settings, 'rejected': {k:dict(v) for k,v in rejected.items()}}
    stage = output.parent/('.'+output.name+'_build'); stage.mkdir(parents=True, exist_ok=True)
    assets = stage/'downloaded_images'; assets.mkdir(exist_ok=True)
    state_path = stage/'state.json'
    fingerprint = object_hash(settings)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'fingerprint': fingerprint, 'selected': {s:[] for s in SOURCES}, 'failed_ids': {s:[] for s in SOURCES}, 'attempts': 0}
    if state['fingerprint'] != fingerprint: raise ValueError('Build inputs changed; choose a different output directory')
    index = DisjointIndex(args.near_distance)
    for logical in ['test', 'dev', 'train']:
        for row in state['selected'][logical]: index.add(row)
    excluded = old_development_exclusions(root)
    started = time.monotonic()
    for logical in ['test', 'dev', 'train']:
        selected = state['selected'][logical]; failed = set(state['failed_ids'][logical])
        present = Counter(r['6_way_label'] for r in selected)
        for label in '012345':
            needed = source_meta[logical]['quota'][label]
            candidates = [r for r in pools[logical] if r['6_way_label'] == label and r['id'] not in failed and r['id'] not in index.ids]
            candidates.sort(key=lambda r: hashlib.sha256(f"{args.seed}:{logical}:{r['id']}".encode()).digest())
            iterator = iter(candidates)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                while present[label] < needed:
                    batch = []
                    while len(batch) < min(args.workers, needed-present[label]):
                        row = next(iterator, None)
                        if row is None: break
                        reason = index.metadata_reason(row)
                        if logical == 'test' and (row['id'] in excluded['ids'] or row['text_sha256'] in excluded['texts']): reason = 'previously_inspected_development'
                        if reason: rejected[logical][reason] += 1; continue
                        batch.append(row)
                    if not batch: raise ValueError(f'Cannot fill {logical} class {label}: {present[label]}/{needed}')
                    outcomes = list(executor.map(lambda r: materialize(r, root, assets, args.download, args.timeout, args.min_side), batch))
                    for row, (path, signature, error, attempted) in zip(batch, outcomes):
                        state['attempts'] += int(attempted)
                        if error == 'dns_unavailable' and not any(p is not None for p,_,_,_ in outcomes) and not network_available():
                            atomic_json(state_path, state)
                            raise ConnectionError('DNS/network unavailable; resume with network access')
                        reason = error or index.metadata_reason(row)
                        if signature:
                            reason = reason or index.image_reason(signature)
                            if logical == 'test' and signature['image_sha256'] in excluded['images']: reason = 'previous_development_image'
                        if reason:
                            rejected[logical][reason] += 1
                            if reason != 'missing_local_image': state['failed_ids'][logical].append(row['id'])
                            continue
                        accepted = {**row, **signature, '_local_image': str(path.relative_to(root))}
                        selected.append(accepted); index.add(accepted); present[label] += 1
                    atomic_json(state_path, state)
                    print('[BUILD]', logical, 'class', label, f'{present[label]}/{needed}',
                          'total', len(selected), 'downloads', state['attempts'], 'seconds', round(time.monotonic()-started), flush=True)
                    if state['attempts'] >= args.max_download_attempts:
                        raise ValueError('Download attempt limit reached; state saved for resume')
    base = yaml.safe_load((root/args.base_config).read_text()); bundle = stage/'bundle'; bundle.mkdir(exist_ok=True)
    for logical in SOURCES:
        folder = bundle/logical; images = folder/'images'; images.mkdir(parents=True, exist_ok=True)
        selected = list(state['selected'][logical]); random.Random(args.seed).shuffle(selected)
        exported = []; samples = []
        for row in selected:
            filename = row['id']+'.jpg'; shutil.copy2(root/row['_local_image'], images/filename)
            clean = {k:row[k] for k in COLUMNS}
            clean.update(image_path=str(output.relative_to(root)/logical/'images'/filename),
                         y_true=row['6_way_label'], split=logical, source_split={'train':'train','dev':'validate','test':'test_public'}[logical],
                         label_source='6_way_label', source_file=source_meta[logical]['path'], source_row=row['source_row'],
                         text_sha256=row['text_sha256'])
            exported.append(clean)
            samples.append({'id':row['id'], 'y_true':row['6_way_label'], 'image_sha256':row['image_sha256'],
                            'pixel_sha256':row['pixel_sha256'], 'text_sha256':row['text_sha256']})
        write_tsv(folder/'fakeddit_sample.tsv', exported)
        support = dict(sorted(Counter(r['6_way_label'] for r in selected).items()))
        manifest = {'status':'ready','dataset':'fakeddit','task':'6way','source_split':exported[0]['source_split'],
                    'label_source':'6_way_label','split':logical,'requested_samples':args.samples,'seed':args.seed,
                    'selected':{'samples':len(selected),'support':support}, 'source':source_meta[logical],
                    'rejected':dict(rejected[logical]), 'samples':samples,
                    'samples_sha256':digest(folder/'fakeddit_sample.tsv'), 'cohort_id':object_hash(samples)[:20]}
        atomic_json(folder/'manifest.json', manifest)
        config = split_config(base, output.relative_to(root), logical, args.samples)
        config['datasets']['fakeddit']['min_text_length'] = args.min_text_length
        (folder/'experiment.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    audit = verify(bundle, args.samples, args.min_side, args.near_distance)
    audit.update(protocol=settings, excluded_previous_development_ids=len(excluded['ids']), downloads=state['attempts'])
    atomic_json(bundle/'audit.json', audit)
    bundle.rename(output)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/processed/fakeddit_splits_v2')
    parser.add_argument('--samples', type=int, default=3000); parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base-config', default='configs/preprocessing/fakeddit_legacy_base.yaml')
    parser.add_argument('--min-text-length', type=int, default=5); parser.add_argument('--min-side', type=int, default=64)
    parser.add_argument('--near-distance', type=int, choices=[-1,0,1,2,3], default=2)
    parser.add_argument('--workers', type=int, default=12); parser.add_argument('--timeout', type=float, default=8)
    parser.add_argument('--max-download-attempts', type=int, default=54000)
    parser.add_argument('--download', action='store_true'); parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.samples < 6 or args.min_side < 1: parser.error('Invalid workers/samples/min-side')
    result = verify(ROOT/args.output,args.samples,args.min_side,args.near_distance) if args.verify else prepare(args)
    print(json.dumps(result if args.inspect else {k:v for k,v in result.items() if k in ['status','splits','total_samples','downloads']},ensure_ascii=False,indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
