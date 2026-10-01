"""Dataset branch switches; pair similarity is diagnostic, not a classifier."""


def relation_settings(task_config, weight):
    flags = {key: task_config.get(key, default) for key, default in
             [('image_label_enabled', True), ('text_label_enabled', True),
              ('image_text_relation_enabled', False)]}
    if not all(isinstance(value, bool) for value in flags.values()):
        raise ValueError('Branch flags must be YAML true/false')
    if not flags['image_label_enabled'] and not flags['text_label_enabled']:
        raise ValueError('Enable at least one label branch; image-text relation only logs similarity, not labels')
    effective = weight if flags['image_label_enabled'] and flags['text_label_enabled'] else (1. if flags['image_label_enabled'] else 0.)
    flags['relation_text_representation'] = ('normalized_mean_chunks' if task_config.get('chunk_long_text', False)
                                             else 'single_text_embedding') if flags['image_text_relation_enabled'] else None
    return effective, flags


def attach_relation(frame, scores):
    if 'image_text_similarity' in scores:
        frame['image_text_similarity'] = scores['image_text_similarity']


def relation_fusion_settings(task_config, dataset):
    import math
    from numbers import Real
    enabled = task_config.get('relation_fusion_enabled', False)
    weight = task_config.get('relation_fusion_weight', .1)
    if not isinstance(enabled, bool):
        raise ValueError('relation_fusion_enabled must be true/false')
    if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('relation_fusion_weight must be finite in [0, 1]')
    if enabled and (dataset != 'fakeddit' or task_config.get('task') != '6way'):
        raise ValueError('Relation fusion supports only Fakeddit task: 6way; no equivalent rule for this task')
    return {'relation_fusion_enabled': enabled, 'relation_fusion_weight': weight}


def fuse_false_connection(probabilities, similarities, labels, weight):
    """Experimental positive evidence for class 2, not a calibrated truth detector.

    d=(1-clip(cosine,-1,1))/2; p'=(p+weight*d*one_hot(2))/(1+weight*d).
    Missing text and zero weight preserve the baseline exactly. No labels fitted.
    """
    import numpy as np
    if len(labels) != 6 or set(labels) != set(range(6)):
        raise ValueError('Relation fusion requires Fakeddit six label IDs 0..5')
    result = probabilities.copy()
    if weight == 0:
        return result
    index = labels.index(2)
    for i, similarity in enumerate(similarities):
        if similarity is None:
            continue
        boost = weight * (1 - np.clip(similarity, -1, 1)) / 2
        result[i, index] += boost
        result[i] /= 1 + boost
    return result
