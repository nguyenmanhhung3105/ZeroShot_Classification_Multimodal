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
