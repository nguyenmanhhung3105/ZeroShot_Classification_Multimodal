"""Quan hệ ảnh–text: policy Fakeddit hoặc đồng thuận CrisisMMD, không fit nhãn thật."""
import math
from numbers import Real

import numpy as np

RELATION_COLUMNS = ("image_text_similarity", "image_text_js_divergence",
                    "image_text_reliability",
                    "image_text_match_evidence", "image_text_mismatch_evidence",
                    "image_text_fusion_gate", "image_text_pair_match_weight", "relation_applied")


def validate_relation_options(options=None):
    """Chuẩn hóa cấu hình; policy lớp chỉ áp dụng cho Fakeddit 6way."""
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("image_text_relation phải là một ánh xạ")
    unknown = set(options) - {"enabled", "mode", "weight", "class_policies"}
    if unknown:
        raise ValueError(f"Khóa image_text_relation không hợp lệ: {sorted(unknown)}")
    enabled = options.get("enabled", False)
    mode = options.get("mode", "diagnostic")
    weight = options.get("weight", 0.25)
    if not isinstance(enabled, bool) or mode not in ("diagnostic", "class_aware", "agreement_fusion", "direct_pair_fusion"):
        raise ValueError("enabled phải true/false; mode phải diagnostic/class_aware/agreement_fusion/direct_pair_fusion")
    if (isinstance(weight, bool) or not isinstance(weight, Real)
            or not math.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError("Trọng số quan hệ phải hữu hạn trong [0, 1]")
    policies = options.get("class_policies", {0: "match", 2: "mismatch"} if mode == "class_aware" else {})
    if not isinstance(policies, dict):
        raise ValueError("class_policies phải là một ánh xạ")
    normalized = {}
    for key, value in policies.items():
        if isinstance(key, bool) or str(key) not in {str(i) for i in range(6)}:
            raise ValueError("Policy chỉ nhận ID Fakeddit 6way từ 0 đến 5")
        label = int(key)
        if label in normalized or value not in ("match", "mismatch", "neutral"):
            raise ValueError("Policy trùng/không hợp lệ")
        if value != "neutral" and (label, value) not in ((0, "match"), (2, "mismatch")):
            raise ValueError("Chỉ True=0 dùng match, False Connection=2 dùng mismatch; lớp khác neutral")
        normalized[label] = value
    if mode in ("agreement_fusion", "direct_pair_fusion") and normalized:
        raise ValueError(f"{mode} không nhận class_policies")
    return {"enabled": enabled, "mode": mode, "weight": float(weight), "class_policies": normalized}


def relation_settings(task_config, dataset, config=None):
    """Flag toàn cục < cấu hình dataset; auto chọn policy theo đúng task."""
    shared = (config or {}).get("inference", {}).get("image_text_relation")
    local = task_config.get("image_text_relation")
    shared = {} if shared is None else shared
    local = {} if local is None else local
    if not isinstance(shared, dict) or not isinstance(local, dict):
        raise ValueError("image_text_relation phải là một ánh xạ")
    merged = {**shared, **local}
    requested_mode = merged.get("mode", "diagnostic")
    if requested_mode == "auto":
        merged["mode"] = "class_aware" if dataset == "fakeddit" and task_config.get("task") == "6way" else "diagnostic"
    options = validate_relation_options(merged)
    if requested_mode == "auto" and options["mode"] == "diagnostic":
        options["class_policies"] = {}
    if options["enabled"] and options["mode"] == "class_aware":
        if dataset != "fakeddit" or task_config.get("task") != "6way":
            raise ValueError("class_aware chỉ dùng Fakeddit 6way; task khác chọn diagnostic")
    if options["enabled"] and options["mode"] in ("agreement_fusion", "direct_pair_fusion"):
        if dataset != "crisismmd" or task_config.get("task") not in ("informativeness", "humanitarian"):
            raise ValueError(f"{options['mode']} chỉ dùng cho hai task CrisisMMD")
    return options, {"relation_enabled": options["enabled"], "relation_mode": options["mode"],
                     "relation_mode_requested": requested_mode,
                     "relation_weight": options["weight"], "relation_class_policies": options["class_policies"],
                     "relation_algorithm": ({"agreement_fusion": "cosine_js_consensus_product_v1",
                                             "direct_pair_fusion": "direct_pair_consensus_v1"}
                                            .get(options["mode"], "cosine_js_confidence_bounded_v1")),
                     "relation_text_representation": ("normalized_mean_chunks" if task_config.get("chunk_long_text", False)
                                                       else "single_text_embedding") if options["enabled"] else None}


def research_relation_config(task_config, settings, dataset_options=None):
    """Cấu hình dataset của base < flag runner nghiên cứu < override dataset."""
    layers = [task_config.get("image_text_relation"), settings.get("image_text_relation"),
              (dataset_options or {}).get("image_text_relation")]
    merged = {}
    for layer in layers:
        if layer is not None:
            if not isinstance(layer, dict):
                raise ValueError("image_text_relation phải là một ánh xạ")
            merged.update(layer)
    return merged


def relation_evidence(image, text, p_image, p_text, empty, *, single_label=True):
    """Cosine không phải xác suất; JS chỉ đo bất đồng nhãn của hai nhánh."""
    image, text = np.asarray(image), np.asarray(text)
    empty = np.asarray(empty, dtype=bool)
    if image.ndim != 2 or image.shape != text.shape or empty.shape != (len(image),):
        raise ValueError("Embedding quan hệ không khớp số mẫu/chiều")
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("Embedding quan hệ phải hữu hạn")
    ni, nt = np.linalg.norm(image, axis=1), np.linalg.norm(text, axis=1)
    valid = ~empty & (ni > 1e-12) & (nt > 1e-12)
    cosine = np.zeros(len(image))
    cosine[valid] = np.clip(np.sum(image[valid] * text[valid], axis=1) / (ni[valid] * nt[valid]), -1, 1)
    result = {"image_text_similarity": [float(r) if ok else None for r, ok in zip(cosine, valid)],
              "relation_applied": [False] * len(image)}
    if single_label:
        pi, pt = np.asarray(p_image, dtype=float), np.asarray(p_text, dtype=float)
        if (pi.ndim != 2 or pi.shape != pt.shape or len(pi) != len(image)
                or not np.isfinite(pi).all() or not np.isfinite(pt).all()
                or np.any(pi < 0) or np.any(pt < 0)
                or not np.allclose(pi.sum(axis=1), 1) or not np.allclose(pt.sum(axis=1), 1)):
            raise ValueError("JS cần hai phân phối single-label chuẩn hóa và hữu hạn")
        mean = (pi + pt) / 2
        def kl(p):
            return np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(mean, 1e-300))), axis=1)
        js = np.clip((kl(pi) + kl(pt)) / (2 * np.log(2)), 0, 1)
        if pi.shape[1] < 2:
            raise ValueError("JS cần ít nhất hai lớp")
        entropy = lambda p: -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1)
        # Hai phân phối cùng phẳng không phải bằng chứng đồng thuận có thông tin.
        reliability = np.clip(1 - np.maximum(entropy(pi), entropy(pt)) / np.log(pi.shape[1]), 0, 1)
        reliability[reliability < 1e-12] = 0
        match = (1 + cosine) / 2 * (1 - js) * reliability
        mismatch = (1 - cosine) / 2 * js * reliability
        for key, values in (("image_text_js_divergence", js), ("image_text_match_evidence", match),
                            ("image_text_mismatch_evidence", mismatch), ("image_text_reliability", reliability)):
            result[key] = [float(x) if ok else None for x, ok in zip(values, valid)]
    return result


def apply_class_relation(probabilities, labels, scores, options, image_weight,
                         *, p_image=None, p_text=None):
    """Fakeddit: hiệu chỉnh theo lớp; CrisisMMD: đồng thuận có điều tiết bởi cặp."""
    result = np.asarray(probabilities).copy()
    if not options["enabled"] or options["mode"] == "diagnostic":
        return result
    if options["mode"] in ("agreement_fusion", "direct_pair_fusion"):
        if not 0 < image_weight < 1 or options["weight"] == 0:
            scores["image_text_fusion_gate"] = [0.0] * len(result)
            if options["mode"] == "direct_pair_fusion":
                scores["image_text_pair_match_weight"] = [None] * len(result)
            return result
        image, text = np.asarray(p_image), np.asarray(p_text)
        if (result.ndim != 2 or result.shape[1] < 2 or image.shape != result.shape
                or text.shape != result.shape or not np.isfinite(result).all()
                or not np.isfinite(image).all() or not np.isfinite(text).all()
                or np.any(result < 0) or np.any(image < 0) or np.any(text < 0)
                or not np.allclose(result.sum(axis=1), 1)
                or not np.allclose(image.sum(axis=1), 1)
                or not np.allclose(text.sum(axis=1), 1)):
            raise ValueError("Fusion quan hệ cần ba phân phối single-label hợp lệ")
        required = (("image_text_similarity", "image_text_js_divergence", "image_text_reliability")
                    if options["mode"] == "agreement_fusion" else ("image_text_similarity",))
        if any(key not in scores for key in required):
            raise ValueError("Fusion quan hệ thiếu điểm ảnh–text")
        similarity = scores["image_text_similarity"]
        if options["mode"] == "direct_pair_fusion":
            if len(similarity) != len(result):
                raise ValueError("Evidence quan hệ không cùng số mẫu")
            valid = np.array([value is not None for value in similarity])
            expected = image_weight * image + (1 - image_weight) * text
            if not np.allclose(result[valid], expected[valid]):
                raise ValueError("Điểm nền phải là fusion của đúng hai nhánh khi có text")
            product = image * text
            normalizer = product.sum(axis=1)
            active = valid & (normalizer > 1e-15)
            match_weight = [None] * len(result)
            for i, cosine in enumerate(similarity):
                if cosine is None:
                    continue
                if not np.isfinite(cosine) or not -1 <= cosine <= 1:
                    raise ValueError("Cosine ảnh–text ngoài miền hợp lệ")
                match_weight[i] = float((1 + cosine) / 2)
            if np.any(active):
                consensus = product[active] / normalizer[active, None]
                strength = np.asarray(match_weight, dtype=object)[active].astype(float)[:, None]
                pair = (1 - strength) / result.shape[1] + strength * consensus
                result[active] = ((1 - options["weight"]) * result[active]
                                  + options["weight"] * pair)
            scores["image_text_fusion_gate"] = np.where(active, options["weight"], 0.0).tolist()
            scores["image_text_pair_match_weight"] = match_weight
            scores["relation_applied"] = active.tolist()
            return result
        disagreement = scores["image_text_js_divergence"]
        reliability = scores["image_text_reliability"]
        if any(len(values) != len(result) for values in (similarity, disagreement, reliability)):
            raise ValueError("Evidence quan hệ không cùng số mẫu")
        valid_relation = np.array([all(value is not None for value in values)
                                   for values in zip(similarity, disagreement, reliability)])
        expected = image_weight * image + (1 - image_weight) * text
        if not np.allclose(result[valid_relation], expected[valid_relation]):
            raise ValueError("Điểm nền phải là fusion của đúng hai nhánh khi có text")
        product = image * text
        normalizer = product.sum(axis=1)
        valid_product = normalizer > 1e-15
        gate = np.zeros(len(result))
        for i, (cosine, js, confidence) in enumerate(zip(similarity, disagreement, reliability)):
            if cosine is None or js is None or confidence is None or not valid_product[i]:
                continue
            if (not np.isfinite(cosine) or not -1 <= cosine <= 1
                    or not np.isfinite(js) or not 0 <= js <= 1
                    or not np.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("Evidence quan hệ ngoài miền hợp lệ")
            gate[i] = options["weight"] * max(0.0, cosine) * (1 - js) * confidence
        active = gate > 0
        if np.any(active):
            consensus = product[active] / normalizer[active, None]
            result[active] = ((1 - gate[active, None]) * result[active]
                              + gate[active, None] * consensus)
        scores["image_text_fusion_gate"] = gate.tolist()
        scores["relation_applied"] = active.tolist()
        return result
    if options["mode"] != "class_aware":
        raise ValueError("Mode quan hệ không hợp lệ")
    if len(labels) != 6 or set(labels) != set(range(6)):
        raise ValueError("class_aware cần đủ sáu ID Fakeddit 0..5")
    if not 0 < image_weight < 1 or options["weight"] == 0:
        return result  # chế độ chỉ ảnh/text và fallback giữ nguyên quyết định nền
    if (result.ndim != 2 or result.shape[1] != 6 or not np.isfinite(result).all()
            or np.any(result < 0) or not np.allclose(result.sum(axis=1), 1)):
        raise ValueError("Điểm nền phải là phân phối sáu lớp")
    evidence = np.zeros_like(result)
    for label, policy in options["class_policies"].items():
        if policy == "neutral":
            continue
        values = scores["image_text_" + policy + "_evidence"]
        if len(values) != len(result) or any(x is not None and (not np.isfinite(x) or not 0 <= x <= 1) for x in values):
            raise ValueError("Evidence quan hệ phải hữu hạn trong [0, 1] hoặc None")
        evidence[:, labels.index(label)] = [0 if x is None else x for x in values]
    corrected = result * (1 + options["weight"] * evidence)
    changed = np.any(evidence > 0, axis=1)
    result[changed] = corrected[changed] / corrected[changed].sum(axis=1, keepdims=True)
    scores["relation_applied"] = changed.tolist()
    return result


def attach_relation(frame, scores):
    for key in RELATION_COLUMNS:
        if key in scores:
            frame[key] = scores[key]
