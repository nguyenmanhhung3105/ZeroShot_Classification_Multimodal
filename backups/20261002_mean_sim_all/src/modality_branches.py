"""Resolve the two label-scoring branches configured per dataset."""


def modality_settings(task_config, image_weight):
    image_enabled = task_config.get("image_label_enabled", True)
    text_enabled = task_config.get("text_label_enabled", True)
    if not isinstance(image_enabled, bool) or not isinstance(text_enabled, bool):
        raise ValueError("image_label_enabled and text_label_enabled must be YAML true/false")
    if not image_enabled and not text_enabled:
        raise ValueError("Enable at least one label-scoring branch")
    effective_weight = image_weight if image_enabled and text_enabled else (1.0 if image_enabled else 0.0)
    return effective_weight, {"image_label_enabled": image_enabled,
                              "text_label_enabled": text_enabled}
