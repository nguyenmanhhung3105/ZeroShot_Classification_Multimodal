"""Name-based top-two group selection and description reranking, without labels."""
import json

import numpy as np
from scipy.special import softmax

from .io import ROOT
from .shared_data import checked_path

METHODS = ["flat_names", "flat_descriptions", "top2_names", "top2_descriptions"]


def validate_spec(spec):
    if spec.get("top_groups") != 2 or spec.get("methods") != METHODS:
        raise ValueError("Rerank v1 requires top-2 groups and all four controls")
    template = spec.get("description_template", "")
    if template.count("{label}") != 1 or template.count("{description}") != 1:
        raise ValueError("Description template must include label and description once")
    template.format(label="example", description="visible characteristics")
    file = checked_path(ROOT / spec["descriptor_file"])
    if not file.is_relative_to(ROOT / "configs/descriptors"):
        raise ValueError("Descriptors must be in configs/descriptors")
    value = json.loads(file.read_text())
    if value.get("version") != 1 or len(value.get("descriptors", {})) != 100:
        raise ValueError("Expected a versioned descriptor bank for 100 classes")
    if any(not isinstance(texts, list) or len(texts) != 2 or
           any(not isinstance(t, str) or not t.strip() for t in texts)
           for texts in value["descriptors"].values()):
        raise ValueError("Each class requires two nonempty descriptions")
    return value


def description_bank(fine_names, spec):
    value = validate_spec(spec)
    descriptors = value["descriptors"]
    if set(descriptors) != set(fine_names):
        raise ValueError("Descriptor names differ from source taxonomy")
    return {str(i): [spec["description_template"].format(label=name.replace("_", " "), description=text)
                     for text in descriptors[name]] for i, name in enumerate(fine_names)}


def predict(fine_logits, description_logits, mapping, top_groups=2):
    """Select groups from name logits only; descriptions cannot change the gate."""
    fine = np.asarray(fine_logits, dtype=np.float64)
    desc = np.asarray(description_logits, dtype=np.float64)
    mapping = np.asarray(mapping)
    if fine.ndim != 2 or desc.shape != fine.shape or not np.isfinite(fine).all() or not np.isfinite(desc).all():
        raise ValueError("Finite aligned name/description logits required")
    if mapping.shape != (fine.shape[1],) or not np.issubdtype(mapping.dtype, np.integer) or np.any(mapping < 0):
        raise ValueError("Invalid group mapping")
    groups = int(mapping.max()) + 1
    if set(mapping.tolist()) != set(range(groups)) or type(top_groups) is not int or not 1 <= top_groups <= groups:
        raise ValueError("Invalid group coverage/top_groups")
    flat = softmax(fine, axis=1)
    mass = np.stack([flat[:, mapping == group].sum(1) for group in range(groups)], axis=1)
    selected = np.argsort(-mass, axis=1, kind="stable")[:, :top_groups]
    allowed = (mapping[None, None, :] == selected[:, :, None]).any(axis=1)
    scores = {"flat_names": flat, "flat_descriptions": softmax(desc, axis=1),
              "top2_names": softmax(np.where(allowed, fine, -np.inf), axis=1),
              "top2_descriptions": softmax(np.where(allowed, desc, -np.inf), axis=1)}
    return scores, mass, selected


def ablation_comparisons(truth, scores):
    truth = np.asarray(truth)
    pairs = [("flat_descriptions", "flat_names"), ("top2_names", "flat_names"),
             ("top2_descriptions", "flat_names"), ("top2_descriptions", "flat_descriptions"),
             ("top2_descriptions", "top2_names")]
    rows = []
    for method, reference in pairs:
        proposed = scores[method].argmax(1) == truth
        baseline = scores[reference].argmax(1) == truth
        rows.append({"method": method, "reference": reference,
                     "accuracy_delta": float(proposed.mean() - baseline.mean()),
                     "corrected": int((proposed & ~baseline).sum()),
                     "harmed": int((~proposed & baseline).sum())})
    return rows
