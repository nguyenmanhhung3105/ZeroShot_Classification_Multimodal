"""Resolve optional text calibration without changing image calibration."""
import math
from numbers import Real


def validate_text_scale(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                              or not math.isfinite(value) or value <= 0):
        raise ValueError("text_logit_scale must be null or a finite positive number")
    return value


def text_scale_settings(config, dataset, vlm, task_config, image_weight):
    from nli_text import validate_backend
    nli = validate_backend(task_config, dataset) == "nli"
    settings = (config or {}).get("inference", {})
    overrides = settings.get("text_logit_scale_overrides") or {}
    requested = validate_text_scale(overrides.get(dataset, settings.get("text_logit_scale")))
    relative = dataset == "mmimdb" and task_config.get("decision_mode", "relative_z") == "relative_z"
    applied = not relative and not nli and image_weight < 1
    effective = (requested if requested is not None else getattr(vlm, "logit_scale", 100.0)) if applied else None
    status = "ignored_nli" if nli else "ignored_relative_z" if relative else "unused_image_only" if not applied else "model_default" if requested is None else "manual"
    metadata = {"text_logit_scale_requested": requested, "text_logit_scale_effective": effective,
                "text_logit_scale_status": status,
                "text_scale_override_applied": applied and requested is not None,
                "fusion_enabled": 0 < image_weight < 1,
                "image_weight": image_weight, "text_weight": 1 - image_weight}
    print(f"[TEXT SCALE] {dataset}: requested={requested}, effective={effective}, status={status}")
    return requested, metadata
