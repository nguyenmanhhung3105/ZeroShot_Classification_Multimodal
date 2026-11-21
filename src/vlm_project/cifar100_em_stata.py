"""Multiclass adaptation of the original CLIP/EM/StatA majority hierarchy."""
import numpy as np
from scipy.special import softmax

from .adaptation import probability_features, em_dirichlet, stata

METHODS = ['flat_clip', 'flat_em', 'flat_stata', 'flat_majority',
           'hierarchy_clip', 'hierarchy_em', 'hierarchy_stata', 'hierarchy_majority']


def validate_spec(spec):
    expected = {'group_size': 75, 'grouping_seeds': [0, 1, 2],
                'em_temperature': 30., 'em_iterations': 20, 'em_mm_iterations': 1000,
                'stata_iterations': 10, 'stata_neighbors': 3, 'methods': METHODS}
    if any(spec.get(key) != value for key, value in expected.items()) or spec['logit_scale'] != 100:
        raise ValueError('EM/StatA v1 preserves the original solver parameters and three seeds')


def majority(votes, classes):
    """Plurality of three voters; all-different ties use the first (CLIP) vote."""
    votes = np.asarray(votes)
    if votes.ndim != 2 or votes.shape[1] != 3 or not np.issubdtype(votes.dtype, np.integer):
        raise ValueError('Expected N x 3 integer votes')
    if np.any(votes < 0) or np.any(votes >= classes):
        raise ValueError('Vote outside taxonomy')
    result = votes[:, 0].copy()
    agree = votes[:, 1] == votes[:, 2]
    result[agree] = votes[agree, 1]
    return result


def restrict(fine_logits, coarse_prediction, mapping):
    """Original hard routing: choose one group then its highest CLIP fine class."""
    mask = mapping[None, :] == coarse_prediction[:, None]
    return np.where(mask, fine_logits, -np.inf).argmax(1)


def predict(features, prototypes, fine_logits, coarse_logits, mapping, spec,
            em_solver=em_dirichlet, stata_solver=stata):
    """No true labels accepted. Flat and coarse branches share identical groups."""
    import torch
    validate_spec(spec)
    x = np.asarray(features)
    mapping = np.asarray(mapping)
    fine_logits, coarse_logits = np.asarray(fine_logits), np.asarray(coarse_logits)
    if fine_logits.shape != (len(x), 100) or coarse_logits.shape != (len(x), 20):
        raise ValueError('Expected 100 fine and 20 coarse classes')
    if mapping.shape != (100,) or not np.array_equal(np.bincount(mapping, minlength=20), np.full(20, 5)):
        raise ValueError('Invalid CIFAR hierarchy')
    if not all(np.isfinite(v).all() for v in [x, fine_logits, coarse_logits]):
        raise ValueError('Nonfinite features/logits')
    fine_clip, coarse_clip = fine_logits.argmax(1), coarse_logits.argmax(1)
    def one_hot(values):
        return np.eye(100, dtype=np.float32)[values]
    scores = {'flat_clip': softmax(fine_logits, axis=1),
              'hierarchy_clip': one_hot(restrict(fine_logits, coarse_clip, mapping))}
    diagnostics = {'tie_rule': 'if three distinct labels, keep CLIP vote',
                   'transductive': 'unlabeled query groups of at most 75; same groups for flat/coarse',
                   'stage2': 'CLIP fine logits restricted to chosen coarse group', 'seeds': {}}
    arrays = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for seed in spec['grouping_seeds']:
            order = np.random.default_rng(seed).permutation(len(x))
            branches = {'fine': {'em': fine_clip.copy(), 'stata': fine_clip.copy()},
                        'coarse': {'em': coarse_clip.copy(), 'stata': coarse_clip.copy()}}
            group_ids = np.empty(len(x), dtype=np.int32)
            solver_details = []
            for group, start in enumerate(range(0, len(x), spec['group_size'])):
                ids = order[start:start+spec['group_size']]
                group_ids[ids] = group
                for level, logits in [('fine', fine_logits), ('coarse', coarse_logits)]:
                    # Preserve EM temperature=30 on cosine, not on scaled logits.
                    query = probability_features(logits[ids] / spec['logit_scale'], spec['em_temperature'])
                    em_labels, detail, _ = em_solver(query[None], iterations=spec['em_iterations'],
                                                     mm_iterations=spec['em_mm_iterations'])
                    branches[level]['em'][ids] = em_labels[0]
                    if len(ids) > spec['stata_neighbors']:
                        _, posterior = stata_solver(x[ids], prototypes[level],
                                                    max_iter=spec['stata_iterations'], neighbors=spec['stata_neighbors'])
                        branches[level]['stata'][ids] = posterior.argmax(1)
                    solver_details.append({'group': group, 'level': level, 'samples': len(ids),
                                           'singleton_fallback': detail.get('singleton_fallback'),
                                           'mm_steps': detail.get('mm_steps'),
                                           'objective_initial': detail['objective'][0] if detail.get('objective') else None,
                                           'objective_final': detail['objective'][-1] if detail.get('objective') else None})
                print(f'EM/StatA seed={seed}: group {group+1}/{(len(x)+74)//75}, N={len(ids)}', flush=True)
            fine_votes = np.stack([fine_clip, branches['fine']['em'], branches['fine']['stata']], axis=1)
            coarse_votes = np.stack([coarse_clip, branches['coarse']['em'], branches['coarse']['stata']], axis=1)
            flat_predictions = {**branches['fine'], 'majority': majority(fine_votes, 100)}
            gates = {**branches['coarse'], 'majority': majority(coarse_votes, 20)}
            for method in ['em', 'stata', 'majority']:
                scores[f'flat_{method}@{seed}'] = one_hot(flat_predictions[method])
                scores[f'hierarchy_{method}@{seed}'] = one_hot(restrict(fine_logits, gates[method], mapping))
            diagnostics['seeds'][str(seed)] = {'solver_groups': solver_details}
            arrays[f'group_ids_{seed}'] = group_ids
            arrays[f'fine_votes_{seed}'] = fine_votes
            arrays[f'coarse_votes_{seed}'] = coarse_votes
    finally:
        torch.set_num_threads(old_threads)
    return scores, softmax(coarse_logits, axis=1), diagnostics, arrays
