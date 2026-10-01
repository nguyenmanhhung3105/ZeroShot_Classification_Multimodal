"""Read-only scoring diagnostics and explicit cohort locks; no fitting or tuning."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _create_lock(path, cohort):
    """Publish a complete JSON atomically, without replacing an existing lock."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".cohort-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(cohort, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_audit(frame, task_config, config, dataset, task, id_col, text_col, label_col,
                  prompts, *, scope="main", limit=None):
    settings = config.get("evaluation", {})
    if not settings.get("enabled", False):
        return None
    if frame.empty:
        raise ValueError("Cannot lock an empty evaluation cohort")
    rows = []
    for _, row in frame.iterrows():
        path = str(row.get("_resolved_image_path", row.get("image_path", "")))
        try:
            stat = Path(path).stat()
            image_stat = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            image_stat = None
        rows.append({"id": str(row[id_col]), "text_sha256": digest(str(row[text_col])),
                     "label": str(row[label_col]), "image_path": path, "image_stat": image_stat})
    cohort = {"dataset": dataset, "task": task, "scope": scope,
              "data_path": str(task_config.get("data_path")),
              "limit": task_config.get("max_samples") if limit is None else limit,
              "min_text_length": task_config.get("min_text_length", 20) if dataset == "fakeddit" else 0,
              "id_col": id_col, "text_col": text_col, "label_col": label_col, "samples": rows}
    fingerprint = digest(cohort)
    mode = settings.get("sample_lock_mode", "off")
    if mode not in ("off", "create_or_verify", "verify"):
        raise ValueError("Unknown sample_lock_mode")
    lock_path = None
    if mode != "off":
        root = Path(settings.get("sample_lock_dir", "results/sample_locks"))
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{dataset}_{task}_{scope}")
        lock_path = root / f"{name}.json"
        if mode == "create_or_verify":
            root.mkdir(parents=True, exist_ok=True)
            _create_lock(lock_path, cohort)
        if not lock_path.exists():
            raise ValueError(f"Sample lock missing: {lock_path}")
        with lock_path.open(encoding="utf-8") as stream:
            saved = json.load(stream)
        if digest(saved) != fingerprint:
            raise ValueError(f"Sample lock mismatch: {lock_path}. Use a NEW sample_lock_dir for a new cohort; do not overwrite the old lock.")
    audit = {"sample_set_id": fingerprint, "cohort": cohort,
             "sample_lock_path": str(lock_path) if lock_path else None,
             "config": config, "effective_dataset": task_config, "prompts": prompts,
             "prompt_id": digest(prompts), "split": settings.get("split", "unverified"),
             "split_verified": False, "truncate_enabled": settings.get("measure_truncation", True)}
    return audit


def token_lengths(vlm, texts):
    """Count before truncation using known OpenCLIP adapters, never infer from padding.

    Unsupported backends report unknown rather than an invented zero rate.
    Clean texts exactly as inference does; inspect only input texts, not prompts.
    """
    tokenizer = getattr(vlm, "tokenizer", None)
    context = getattr(getattr(vlm, "model", None), "context_length", None)
    lengths = []
    try:
        name = type(tokenizer).__name__
        if not isinstance(context, int) or context < 1:
            raise ValueError("Context length unavailable")
        if name == "SimpleTokenizer" and not getattr(tokenizer, "reduction_fn", None):
            lengths = [len(tokenizer.encode(t.strip())) + 2 if isinstance(t, str) and t.strip() else None for t in texts]
        elif name == "HFTokenizer" and not getattr(tokenizer, "tokenizer_mode", ""):
            lengths = [len(tokenizer.tokenizer.encode(tokenizer.clean_fn(t.strip()),
                        add_special_tokens=True, truncation=False)) if isinstance(t, str) and t.strip() else None for t in texts]
        else:
            raise ValueError(f"Unsupported tokenizer: {name}")
        return lengths, context, "measured"
    except (AttributeError, TypeError, ValueError, NotImplementedError) as exc:
        warnings.warn(f"Truncation audit unavailable: {exc}")
        return [None] * len(texts), context, "unknown"


def finish_audit(audit, vlm, frame, texts, truth, labels, fused, image, text, empty,
                 weight, *, multilabel=False, decision_meta=None, fallback=True):
    if audit is None:
        return {}
    meta = decision_meta or {}
    audit["effective_inference"] = {**meta, "image_weight": weight, "fallback_top1": fallback,
                                    "logit_scale": getattr(vlm, "logit_scale", None),
                                    "logit_bias": getattr(vlm, "logit_bias", None)}
    relative = meta.get("score_type") == "relative_z"
    cutoff = meta.get("relative_z_threshold", 1.0) if relative else meta.get("threshold", 0.22)
    empty = np.asarray(empty, dtype=bool)
    if (len(truth) != len(texts) or len(frame) != len(texts)
            or len(audit["cohort"]["samples"]) != len(texts) or empty.shape != (len(texts),)):
        raise ValueError("Audit sample alignment mismatch")
    for scores in (fused, image, text):
        values = np.asarray(scores)
        if values.shape != (len(texts), len(labels)) or not np.isfinite(values).all():
            raise ValueError("Audit scores must be finite and match samples/labels")
    report, summary = [], {"sample_set_id": audit["sample_set_id"], "prompt_id": audit["prompt_id"],
                           "split": audit["split"]}
    for branch, scores in (("image", image), ("text", text), ("fusion", fused)):
        # Never report an unencoded branch or image fallback as text-only evidence.
        mask = (np.ones(len(texts), bool) if branch == "fusion" else
                (~empty & (weight < 1)) if branch == "text" else
                (np.ones(len(texts), bool) if weight > 0 else empty))
        scores = np.asarray(scores)[mask]
        actual = [t for t, keep in zip(truth, mask) if keep]
        row = {"branch": branch, "n_samples": int(mask.sum()), "score_type": "relative_z" if relative else "probability"}
        summary[f"{branch}_n_samples"] = row["n_samples"]
        if len(scores):
            if multilabel:
                unknown = {label for values in actual for label in values} - set(labels)
                if unknown:
                    raise ValueError(f"Unknown labels in audit: {unknown}")
                selected = scores > cutoff if relative else scores >= cutoff
                missing = ~selected.any(axis=1)
                if fallback:
                    selected[np.flatnonzero(missing), scores[missing].argmax(axis=1)] = True
                target = np.array([[label in values for label in labels] for values in actual], dtype=int)
                metrics = {f"{avg}_f1": float(f1_score(target, selected, average=avg, zero_division=0))
                           for avg in ("micro", "macro", "samples")}
                row.update(mean_predicted_labels=float(selected.sum(axis=1).mean()),
                           fallback_rate=float(missing.mean()) if fallback else 0.0)
            else:
                if set(actual) - set(labels):
                    raise ValueError("Unknown single-label class in audit")
                predicted = [labels[i] for i in scores.argmax(axis=1)]
                metrics = {"macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
                           "balanced_accuracy": float(balanced_accuracy_score(actual, predicted))}
            row.update(metrics)
            summary.update({f"{branch}_{k}": v for k, v in metrics.items()})
            ordered = np.sort(scores, axis=1)
            row["top2_margin"] = float((ordered[:, -1] - ordered[:, -2]).mean())
            if not relative:
                p = np.clip(scores.astype(float), 1e-12, 1 - 1e-12)
                entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(axis=1) if multilabel else -(p * np.log(p)).sum(axis=1)
                row["entropy"] = float(entropy.mean())
            row.update(audit.get("cosine", {}).get(branch, {}))
        report.append(row)
    if audit["truncate_enabled"]:
        lengths, context, status = token_lengths(vlm, texts)
    else:
        lengths, context, status = [None] * len(texts), None, "disabled"
    measured = [n for n in lengths if n is not None]
    truncated = sum(n > context for n in measured) if measured else 0
    audit["truncation"] = {"status": status, "context_length": context, "n_nonempty": int((~empty).sum()),
                           "n_measured": len(measured), "n_truncated": truncated if measured else None,
                           "truncated_rate": truncated / len(measured) if measured else None}
    # Keep the rate visible next to the text-branch diagnostics. It measures all
    # nonempty inputs, including when the text encoder is disabled by weight=1.
    report[1].update({f"input_{key}": value for key, value in audit["truncation"].items()})
    if truncated:
        print(f"[TRUNCATE] {truncated}/{len(measured)} texts exceed context_length={context}")
    audit["text_lengths"] = [{"sample_index": i, "id": audit["cohort"]["samples"][i]["id"], "token_count": n,
                             "truncated": n > context if n is not None else None} for i, n in enumerate(lengths)]
    audit["diagnostics"] = report
    audit["summary"] = summary
    audit["model"] = vlm.model_name
    audit["config_id"] = digest({"config": audit["config"], "dataset": audit["effective_dataset"],
                                 "prompts": audit["prompts"], "inference": audit["effective_inference"],
                                 "model": vlm.model_name})
    summary["config_id"] = audit["config_id"]
    print(pd.DataFrame(report).to_string(index=False))
    return summary


def save_audit(audit, run_dir):
    if not audit:
        return
    root = Path(run_dir)
    source = Path(__file__).resolve().parent
    audit["source_sha256"] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source.rglob("*.py") if "__pycache__" not in p.parts}
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source.parent,
                             capture_output=True, text=True, timeout=5)
        audit["git_hash"] = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        audit["git_hash"] = None
    pd.DataFrame(audit["diagnostics"]).to_csv(root / "diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame(audit["text_lengths"]).to_csv(root / "text_lengths.tsv", sep="\t", index=False)
    with (root / "evaluation_snapshot.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, default=str)
