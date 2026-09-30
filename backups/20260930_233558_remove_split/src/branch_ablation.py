"""Small paired ablation. No fitting, test statistics adaptation, or prompt generation.

Keep original prompt wording: only route fixed subsets to image/text branches.
Inference APIs and all legacy entry points keep their default behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from inference import (build_class_embeddings, encode_multimodal_embeddings, relative_multilabel_scores,
                       predict_single_label, predict_multi_label)
from prompts import fakeddit_prompts, crisismmd_prompts, mmimdb_prompts


def route_prompts(prompts, routes, dataset, task):
    """Fixed index routing; label order always comes from legacy prompts."""
    try:
        spec = routes[dataset][task]
    except KeyError as exc:
        raise ValueError(f"No reviewed branch routes for {dataset}/{task}") from exc
    result = {}
    for branch in ("image", "text"):
        selection = spec[branch]
        if isinstance(selection, dict) and set(selection) != {str(k) for k in prompts}:
            raise ValueError("Route labels must match the complete legacy label set")
        result[branch] = {}
        for label, sentences in prompts.items():
            indices = selection[str(label)] if isinstance(selection, dict) else selection
            if (not indices or len(set(indices)) != len(indices)
                    or any(type(i) is not int or i < 0 or i >= len(sentences) for i in indices)):
                raise ValueError(f"Invalid route for {branch}/{label}")
            result[branch][label] = [sentences[i] for i in indices]
    return result


def decisions(scores, labels, multilabel=False, threshold=0.22, fallback=True, strict=False):
    if not multilabel:
        return [labels[i] for i in scores.argmax(axis=1)]
    predictions = []
    for row in scores:
        chosen = [label for label, score in zip(labels, row)
                  if (score > threshold if strict else score >= threshold)]
        if not chosen and fallback:
            chosen = [labels[int(row.argmax())]]
        predictions.append(chosen)
    return predictions


def metrics(truth, predicted, labels, multilabel):
    if multilabel:
        def binary(rows):
            unknown = {label for row in rows for label in row} - set(labels)
            if unknown:
                raise ValueError(f"Unknown labels: {unknown}")
            return np.array([[label in row for label in labels] for row in rows], dtype=int)
        truth, predicted = binary(truth), binary(predicted)
        return {f"{average}_f1": f1_score(truth, predicted, average=average, zero_division=0)
                for average in ("micro", "macro", "samples")}
    if (set(truth) | set(predicted)) - set(labels):
        raise ValueError("Unknown single-label class")
    return {
        "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "accuracy": accuracy_score(truth, predicted),
    }


def diagnostics(scores, multilabel):
    p = np.clip(scores.astype(float), 1e-12, 1 - 1e-12)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(axis=1) if multilabel else -(p * np.log(p)).sum(axis=1)
    ordered = np.sort(scores, axis=1)
    return {"entropy": float(entropy.mean()),
            "top2_margin": float((ordered[:, -1] - ordered[:, -2]).mean())}


def compare(vlm, images, texts, truth, prompts, branches=None, *, image_weight=0.5,
            batch_size=4, multilabel=False, threshold=0.22, fallback=True,
            show_progress=True, decision_mode="fixed_threshold", relative_z_threshold=1.0):
    """Encode inputs once; compare both variants on exactly the same paired rows."""
    if not 0 < image_weight < 1:
        raise ValueError("Ablation requires both modalities: 0 < image_weight < 1")
    if len(truth) != len(texts):
        raise ValueError("Truth and input lengths differ")
    if decision_mode not in ("fixed_threshold", "relative_z"):
        raise ValueError("Unknown decision_mode")
    if not np.isfinite(relative_z_threshold) or relative_z_threshold < 0:
        raise ValueError("relative_z_threshold must be finite and nonnegative")
    cache = encode_multimodal_embeddings(vlm, images, texts, batch_size=batch_size,
                                         show_progress=show_progress, image_weight=None)
    legacy, labels = build_class_embeddings(vlm, prompts)
    variants = [("baseline", None, None)]
    if branches is not None:
        matrices = []
        for name in ("image", "text"):
            if list(branches[name]) != labels:
                raise ValueError("Branch labels/order must equal baseline labels/order")
            matrix, _ = build_class_embeddings(vlm, branches[name])
            if not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5):
                raise ValueError("Degenerate class prototype")
            matrices.append(matrix)
        variants.append(("split_prompts", *matrices))
    rows, predictions = [], []
    predictor = predict_multi_label if multilabel else predict_single_label
    options = {"threshold": threshold, "fallback_top1": fallback} if multilabel else {}
    # Run only the selected decision rule, for shared and split prompts.
    experiments = [(name, im, tx, False) for name, im, tx in variants]
    if multilabel and decision_mode == "relative_z":
        experiments = [("relative_z_" + name, im, tx, True) for name, im, tx in variants]
    for name, image_classes, text_classes, relative in experiments:
        cutoff = relative_z_threshold if relative else threshold
        if relative:
            cosine_image = cache[0] @ (legacy if image_classes is None else image_classes).T
            cosine_text = cache[1] @ (legacy if text_classes is None else text_classes).T
            fused, pi, pt = relative_multilabel_scores(cosine_image, cosine_text, image_weight, cache[2])
            output = (None, fused, pi, pt)
        else:
            output = predictor(vlm, images, texts, legacy, labels, image_weight=image_weight,
                           batch_size=batch_size, show_progress=False, encoded_embeddings=cache,
                           img_class_embeds=image_classes, txt_class_embeds=text_classes, **options)
        row = {"variant": name, "split_prompts": image_classes is not None,
               "decision_mode": "relative_z" if relative else "fixed_threshold" if multilabel else "argmax",
               "score_type": "relative_z" if relative else "probability",
               "threshold": threshold if multilabel and not relative else None,
               "relative_z_threshold": cutoff if relative else None,
               "apply_bias_to_text": not relative,
               "model_scale_applied": not relative,
               "n_samples": len(texts), "n_empty_text": int(cache[2].sum())}
        for branch, scores, matrix in (("fusion", output[1], None),
                                      ("image", output[2], image_classes),
                                      ("text", output[3], text_classes)):
            # Text-only with absent text is undefined, not secretly image fallback.
            mask = ~cache[2] if branch == "text" else np.ones(len(texts), dtype=bool)
            row[f"{branch}_n_samples"] = int(mask.sum())
            selected = scores[mask]
            pred = decisions(selected, labels, multilabel, cutoff, fallback, strict=relative)
            selected_truth = [y for y, keep in zip(truth, mask) if keep]
            if len(selected):
                row.update({f"{branch}_{key}": value for key, value in
                            metrics(selected_truth, pred, labels, multilabel).items()})
                if relative:
                    ordered = np.sort(selected, axis=1)
                    row[f"{branch}_top2_margin"] = float((ordered[:, -1] - ordered[:, -2]).mean())
                else:
                    row.update({f"{branch}_{key}": value for key, value in diagnostics(selected, multilabel).items()})
                if branch != "fusion":
                    embeds = cache[0] if branch == "image" else cache[1]
                    cosine = embeds[mask] @ (legacy if matrix is None else matrix).T
                    row[f"{branch}_cosine_mean"] = float(cosine.mean())
                    row[f"{branch}_cosine_std"] = float(cosine.std())
                if multilabel:
                    row[f"{branch}_mean_predicted_labels"] = float(np.mean([len(y) for y in pred]))
                    accepted = selected > cutoff if relative else selected >= cutoff
                    row[f"{branch}_fallback_rate"] = float(np.mean(~accepted.any(axis=1))) if fallback else 0.0
            for index, y_pred, score in zip(np.flatnonzero(mask), pred, selected):
                predictions.append({"variant": name, "branch": branch, "sample_index": int(index),
                                    "score_type": row["score_type"],
                                    "y_true": truth[index], "y_pred": y_pred, "scores": score.tolist()})
        rows.append(row)
    return rows, predictions


def run(config_path):
    # Delayed imports: --help and synthetic tests need no pretrained backend.
    from run import common
    from models.model_registry import load_model
    with open(config_path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    split = cfg.get("split", "exploratory")
    if split not in ("exploratory", "val"):
        raise ValueError("This initial comparison runner does not run test; use exploratory or verified val")
    if split == "val" and not cfg.get("data_path"):
        raise ValueError("val requires an explicit, independently verified data_path")
    limit = cfg.get("max_samples", 5)
    if type(limit) is not int or limit < 1:
        raise ValueError("Set a positive explicit max_samples")
    base = common.load_config(cfg["base_config"])
    dataset, task = cfg["dataset"], cfg["task"]
    task_cfg = dict(base["datasets"][dataset], task=task, max_samples=limit)
    task_cfg["data_path"] = cfg.get("data_path", task_cfg["data_path"])
    if dataset == "fakeddit":
        prompts = fakeddit_prompts.get_prompt_set(task)
        column = task_cfg.get(f"label_col_{task}", "6_way_label" if task == "6way" else "2_way_label")
    elif dataset == "crisismmd":
        prompts = crisismmd_prompts.get_prompt_set(task)
        column = common.crisismmd_label_column(task_cfg)
    elif dataset == "mmimdb" and task == "genres":
        prompts = mmimdb_prompts.get_prompt_set()
        column = task_cfg.get("label_col", "genres")
    else:
        raise ValueError("Unsupported dataset/task")
    branches = None
    routes = None
    if cfg.get("split_prompts", False):
        with open(cfg["prompt_routes"], encoding="utf-8") as stream:
            routes = json.load(stream)
        branches = route_prompts(prompts, routes, dataset, task)
    text_col, id_col = task_cfg["text_col"], task_cfg["id_col"]
    min_length = cfg.get("min_text_length", 20) if dataset == "fakeddit" else 0
    frame = common.load_dataframe(task_cfg["data_path"], max_samples=limit,
                                  text_col=text_col, min_text_length=min_length)
    common.validate_columns(frame, [id_col, text_col, column], dataset)
    paths = common.resolve_image_paths(frame, task_cfg)
    images, indices = common.load_images_safe(paths)
    skipped = len(frame) - len(indices)
    frame = frame.iloc[indices].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid paired samples")
    texts = frame[text_col].fillna("").astype(str).tolist()
    values = frame[column].tolist()
    if dataset == "fakeddit":
        truth = fakeddit_prompts.normalize_labels(values, task)
    elif dataset == "crisismmd":
        truth = common.normalize_single_labels(values, crisismmd_prompts.get_label_names(task), list(prompts))
    else:
        truth = [common.parse_multilabel(v) for v in values]
    warnings.warn("Exploratory comparison is not evidence of held-out improvement; split provenance must be checked")
    output = Path(cfg.get("output_dir", "results/ablation"))
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{dataset}_{task}_", dir=output))
    def write_json(name, value):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot = {"ablation": cfg, "base": base, "effective_dataset": task_cfg,
                "min_text_length": min_length, "legacy_prompts": prompts,
                "branch_prompts": branches, "routes": routes}
    config_id = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()[:16]
    write_json("config_and_prompts.json", snapshot)
    # Hash source files only, not datasets, credentials, weights or git diffs.
    source_root = Path(__file__).resolve().parent
    sources = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in source_root.rglob("*.py") if "__pycache__" not in p.parts}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    write_json("provenance.json", {"git_hash": git.stdout.strip() if git.returncode == 0 else None,
                                  "source_sha256": sources, "config_id": config_id,
                                  "n_skipped_images": skipped, "split_verified_by_runner": False})
    write_json("samples.json", [{"sample_index": i, "id": str(row[id_col]),
                                 "image_path": images.paths[i],
                                 "text_sha256": hashlib.sha256(texts[i].encode()).hexdigest(),
                                 "y_true": truth[i]} for i, (_, row) in enumerate(frame.iterrows())])
    results = []
    for model_name in cfg.get("models", base["models"]):
        threshold = common.resolve_override(task_cfg.get("threshold", 0.22), task_cfg.get("thresholds"), model_name)
        vlm = load_model(model_name, device=base.get("device"))
        try:
            rows, predicted = compare(vlm, images, texts, truth, prompts, branches,
                                      image_weight=cfg.get("image_weight", 0.5),
                                      batch_size=cfg.get("batch_size", 4), multilabel=dataset == "mmimdb",
                                      threshold=threshold, fallback=task_cfg.get("fallback_top1", True),
                                      decision_mode=cfg.get("decision_mode", "fixed_threshold"),
                                      relative_z_threshold=cfg.get("relative_z_threshold", 1.0))
            for row in rows:
                row.update(dataset=dataset, task=task, model=model_name, config_id=config_id,
                           split=split, protocol="fixed_no_fit", prompt_provenance="legacy_exploratory",
                           image_weight=cfg.get("image_weight", 0.5),
                           logit_scale=vlm.logit_scale, logit_bias=vlm.logit_bias,
                           n_skipped_images=skipped)
            results.extend(rows)
            write_json(f"predictions_{model_name}.json", predicted)
            pd.DataFrame(results).to_csv(directory / "results.tsv", sep="\t", index=False)
        finally:
            vlm.unload()
    print(pd.DataFrame(results).to_string(index=False))
    print(f"[OUTPUT] {directory}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ablation_config.yaml")
    run(parser.parse_args().config)
