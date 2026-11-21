"""Controlled CIFAR-100 flat/hard/soft hierarchy experiment; frozen MetaCLIP2."""
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

from .backbone import Backbone
from .evaluate import metric, publish, table
from .io import ROOT, MODEL, digest, new_run, sha256, write_json
from .shared_data import checked_path, unique_file

METHODS = ["flat", "hierarchy_hard", "hierarchy_soft"]
SOURCE_FILES = ["cifar100_two_stage.py", "backbone.py", "io.py", "evaluate.py", "shared_data.py"]


def specification(file=None):
    file = checked_path(file or ROOT / "configs/cifar100_two_stage.json")
    spec = json.loads(file.read_text())
    rerank = spec.get("variant") == "candidate_rerank"
    expected_methods = METHODS
    if rerank:
        from .cifar100_rerank import METHODS as expected_methods, validate_spec
        validate_spec(spec)
    elif spec.get("variant") == "em_stata":
        from .cifar100_em_stata import METHODS as expected_methods, validate_spec
        validate_spec(spec)
    elif spec.get("variant") is not None:
        raise ValueError("Unknown CIFAR experiment variant")
    if spec.get("version") != 1 or spec.get("model") != MODEL or spec.get("methods") != expected_methods:
        raise ValueError("Unsupported CIFAR experiment configuration")
    if spec.get("dataset_id") != "nguyenhung3105/vlm-cifar100-sources-v1":
        raise ValueError("Unexpected CIFAR dataset ID")
    for key in ["seed", "dev_samples", "batch_size"]:
        if type(spec.get(key)) is not int:
            raise ValueError(f"Expected integer {key}")
    if spec["seed"] < 0 or not 1 <= spec["dev_samples"] < 50000 or not 1 <= spec["batch_size"] <= 64:
        raise ValueError("Invalid seed/dev size/batch size")
    if not np.isfinite(spec["logit_scale"]) or spec["logit_scale"] <= 0:
        raise ValueError("Invalid logit scale")
    if spec["prompt_template"].count("{label}") != 1:
        raise ValueError("Prompt template must contain {label} exactly once")
    spec["prompt_template"].format(label="example")
    return spec


def split_indices(spec, split, smoke=False):
    if split not in ["dev", "test"] or (split == "test" and smoke):
        raise ValueError("Only dev smoke is allowed; test requires full evaluation")
    if split == "dev":
        # Pick IDs before opening labels or pixels; no fitting on remaining train.
        ids = np.sort(np.random.default_rng(spec["seed"]).permutation(50000)[:spec["dev_samples"]])
    else:
        ids = np.arange(10000)
    return ids[:5] if smoke else ids


def locate_source(input_root="/kaggle/input"):
    matches = []
    for file in checked_path(input_root).rglob("download_manifest.json"):
        value = json.loads(checked_path(file).read_text())
        if value.get("dataset") == "cifar100":
            matches.append(file.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected one CIFAR source, found {matches}")
    return matches[0]


def taxonomy(root):
    metadata = json.loads(checked_path(root / "download_manifest.json").read_text())
    if metadata.get("dataset") != "cifar100" or metadata.get("status") != "sources_verified":
        raise ValueError("CIFAR source is not verified")
    if metadata.get("splits") != {"train": 50000, "test": 10000}:
        raise ValueError("Unexpected CIFAR source splits")
    fine = metadata["class_names"]["fine"]
    coarse = metadata["class_names"]["coarse"]
    mapping = np.asarray([metadata["fine_to_coarse"][str(i)] for i in range(100)])
    if len(fine) != 100 or len(coarse) != 20 or len(set(fine)) != 100 or len(set(coarse)) != 20:
        raise ValueError("CIFAR class names incomplete or duplicated")
    if not np.issubdtype(mapping.dtype, np.integer) or np.any(mapping < 0) or np.any(mapping >= 20):
        raise ValueError("Invalid fine-to-coarse IDs")
    if not np.array_equal(np.bincount(mapping, minlength=20), np.full(20, 5)):
        raise ValueError("Each coarse class must have five children")
    return metadata, fine, coarse, mapping


def predict(fine_logits, coarse_logits, mapping):
    """No ground truth accepted. All variants share the same fine logits."""
    fine_logits = np.asarray(fine_logits, dtype=np.float64)
    coarse_logits = np.asarray(coarse_logits, dtype=np.float64)
    mapping = np.asarray(mapping)
    if fine_logits.ndim != 2 or coarse_logits.ndim != 2 or len(fine_logits) != len(coarse_logits):
        raise ValueError("Logit shapes must align")
    if mapping.shape != (fine_logits.shape[1],) or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("Mapping shape/dtype mismatch")
    if not np.isfinite(fine_logits).all() or not np.isfinite(coarse_logits).all():
        raise ValueError("Nonfinite logits")
    groups = coarse_logits.shape[1]
    if np.any(mapping < 0) or np.any(mapping >= groups) or set(mapping.tolist()) != set(range(groups)):
        raise ValueError("Mapping must cover all groups")
    conditional = np.zeros_like(fine_logits)
    for group in range(groups):
        children = np.flatnonzero(mapping == group)
        conditional[:, children] = softmax(fine_logits[:, children], axis=1)
    coarse = softmax(coarse_logits, axis=1)
    gate = coarse.argmax(1)
    hard = conditional * (mapping[None, :] == gate[:, None])
    soft = conditional * coarse[:, mapping]
    return {"flat": softmax(fine_logits, axis=1), "hierarchy_hard": hard,
            "hierarchy_soft": soft}, coarse


def protocol(spec, root):
    libraries = {}
    for name in ["numpy", "scipy", "scikit-learn", "Pillow", "torch", "open_clip_torch", "transformers"]:
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    value = {"specification": spec,
            "code": {name: sha256(Path(__file__).parent / name) for name in SOURCE_FILES},
            "requirements_sha256": sha256(ROOT / "requirements.txt"), "libraries": libraries,
            "source_manifest_sha256": sha256(root / "download_manifest.json"),
            "split_rule": "sorted(default_rng(seed).permutation(50000)[:dev_samples]); no labels used",
            "fine_prototypes": "one prompt per class; unit normalized",
            "coarse_prototypes": "direct prompts of official 20 superclass names",
            "soft_rule": "P(group|image) * softmax(fine_logits restricted to group's children)",
            "tie_rule": "first class/group in official ID order"}
    if spec.get("variant") == "candidate_rerank":
        value["code"]["cifar100_rerank.py"] = sha256(Path(__file__).parent / "cifar100_rerank.py")
        value.update(descriptor_sha256=sha256(ROOT / spec["descriptor_file"]),
                     coarse_prototypes=None, soft_rule=None,
                     routing="sum flat-name probabilities per group, stable top-2",
                     description_prototypes="mean of two unit prompt embeddings, then unit normalized")
    if spec.get("variant") == "em_stata":
        for name in ['cifar100_em_stata.py', 'adaptation.py']:
            value['code'][name] = sha256(Path(__file__).parent / name)
        value.update(variant='em_stata', transductive='same unlabeled groups of 75, seeds 0/1/2',
                     majority_tie='all different -> CLIP vote', soft_rule=None,
                     stage2='CLIP fine scores restricted to selected group; no fine expert retraining')
    return value


def validate_dev(dev_run, current_protocol):
    if not dev_run:
        raise ValueError("Test requires --dev-run from a completed full dev run")
    folder = checked_path(dev_run)
    completion = json.loads((folder / "completed.json").read_text())
    run = json.loads((folder / "run.json").read_text())
    if completion.get("status") != "complete" or completion.get("run_sha256") != sha256(folder / "run.json"):
        raise ValueError("Dev completion checksum mismatch")
    if run.get("status") != "complete" or run.get("split") != "dev" or run.get("mode") != "full":
        raise ValueError("A smoke run cannot authorize test")
    if run.get("protocol") != current_protocol or run.get("samples") != current_protocol["specification"]["dev_samples"]:
        raise ValueError("Dev protocol differs; rerun dev before test")
    for name, checksum in run["output_sha256"].items():
        if checked_path(folder / name).parent != folder or sha256(folder / name) != checksum:
            raise ValueError("Dev artifact checksum mismatch")
    return {"dev_run_sha256": sha256(folder / "run.json")}


def run(root, output_root="results/final/cifar100_two_stage", *, split="dev", smoke=True,
        dev_run=None, spec_file=None, device="auto", backend_factory=Backbone):
    from PIL import Image

    spec = specification(spec_file)
    rerank = spec.get("variant") == "candidate_rerank"
    adapted = spec.get("variant") == "em_stata"
    ids = split_indices(spec, split, smoke)
    root = checked_path(root)
    metadata, fine, coarse, mapping = taxonomy(root)
    current_protocol = protocol(spec, root)
    test_lock = validate_dev(dev_run, current_protocol) if split == "test" else None
    binary_split = "train" if split == "dev" else "test"
    binary = unique_file(root, f"{binary_split}.bin")
    if binary.stat().st_size != metadata["splits"][binary_split] * 3074:
        raise ValueError("CIFAR binary length mismatch")
    output = new_run({"output_root": output_root}, "cifar100", split, smoke)
    banks = {level: {str(i): [spec["prompt_template"].format(label=name.replace("_", " "))]
                     for i, name in enumerate(names)} for level, names in [("fine", fine), ("coarse", coarse)]}
    if rerank:
        from .cifar100_rerank import description_bank
        banks = {"fine": banks["fine"], "descriptions": description_bank(fine, spec)}
    plan = {"kind": "cifar100_em_stata" if adapted else "cifar100_rerank" if rerank else "cifar100_two_stage", "model": MODEL, "dataset": "cifar100", "split": split, "mode": "smoke" if smoke else "full",
            "samples": len(ids), "protocol": current_protocol, "test_lock": test_lock,
            "classes": {"fine": fine, "coarse": coarse}, "prompts": banks,
            "cohort_ids_sha256": digest(ids.tolist()), "smoke_quality_claim": False,
            "scope": "Frozen MetaCLIP2; fixed parameters, no fitting; not a paper reproduction"}
    if adapted:
        plan['scope'] = 'Transductive EM/StatA on unlabeled query groups; no labeled fine-tuning; multiclass adaptation of Fakeddit majority'
    write_json(output / "frozen_plan.json", plan)
    write_json(output / "run.json", {**plan, "status": "running"})
    backend = backend_factory(device)
    started = time.perf_counter()
    feature_blocks, labels, parents = [], [], []
    record_hash = hashlib.sha256()
    try:
        prototypes = {level: backend.prototypes(bank) for level, bank in banks.items()}
        for level, values in prototypes.items():
            norm = np.linalg.norm(values, axis=1, keepdims=True)
            if not np.isfinite(values).all() or np.any(norm <= 0):
                raise ValueError("Invalid prototype")
            prototypes[level] = values / norm
        with binary.open("rb") as stream:
            for start in range(0, len(ids), spec["batch_size"]):
                images = []
                try:
                    for index in ids[start:start+spec["batch_size"]]:
                        stream.seek(int(index) * 3074)
                        record = stream.read(3074)
                        if len(record) != 3074:
                            raise ValueError("Truncated record")
                        parent, label = record[:2]
                        if not 0 <= label < 100 or parent != mapping[label]:
                            raise ValueError("Record taxonomy mismatch")
                        labels.append(label)
                        parents.append(parent)
                        record_hash.update(int(index).to_bytes(8, "little") + record)
                        planes = [Image.frombytes("L", (32, 32), record[2+i*1024:2+(i+1)*1024]) for i in range(3)]
                        images.append(Image.merge("RGB", planes))
                    features = np.asarray(backend.encode_images(images))
                    if features.ndim != 2 or len(features) != len(images) or not np.isfinite(features).all():
                        raise ValueError("Invalid image features")
                    norm = np.linalg.norm(features, axis=1, keepdims=True)
                    if np.any(norm <= 0):
                        raise ValueError("Zero image feature")
                    feature_blocks.append(features / norm)
                finally:
                    for image in images:
                        image.close()
                done = min(start + spec["batch_size"], len(ids))
                if start == 0 or done % 128 == 0 or done == len(ids):
                    print(f"CIFAR {split}: encoded {done}/{len(ids)}", flush=True)
        features = np.concatenate(feature_blocks)
        plan["model_runtime"] = getattr(backend, "metadata", {"backend": type(backend).__name__})
    finally:
        backend.unload()
    fine_logits = spec["logit_scale"] * (features @ prototypes["fine"].T)
    if adapted:
        from .cifar100_em_stata import predict as adapted_predict
        coarse_logits = spec["logit_scale"] * (features @ prototypes["coarse"].T)
        scores, coarse_prob, adaptation_details, adaptation_arrays = adapted_predict(
            features, prototypes, fine_logits, coarse_logits, mapping, spec)
        write_json(output / 'adaptation_details.json', adaptation_details)
        np.savez_compressed(output / 'adaptation_groups.npz', **adaptation_arrays)
        replay_extra = {'coarse_logits': coarse_logits, 'image_features': features,
                        'fine_prototypes': prototypes['fine'], 'coarse_prototypes': prototypes['coarse']}
    elif rerank:
        from .cifar100_rerank import predict as rerank_predict
        description_logits = spec["logit_scale"] * (features @ prototypes["descriptions"].T)
        scores, coarse_prob, candidate_groups = rerank_predict(fine_logits, description_logits, mapping, spec["top_groups"])
        replay_extra = {"description_logits": description_logits, "candidate_groups": candidate_groups,
                        "image_features": features}
    else:
        coarse_logits = spec["logit_scale"] * (features @ prototypes["coarse"].T)
        scores, coarse_prob = predict(fine_logits, coarse_logits, mapping)
        replay_extra = {"coarse_logits": coarse_logits}
    truth, parent_truth = np.asarray(labels), np.asarray(parents)
    records, predictions, details, comparisons = [], [], {}, []
    baseline = scores["flat_clip" if adapted else "flat_names" if rerank else "flat"].argmax(1)
    for tag, probabilities in scores.items():
        method, seed = tag.split('@') if adapted and '@' in tag else (tag, None)
        seed = int(seed) if seed is not None else None
        predicted = probabilities.argmax(1)
        values = metric(truth, predicted, list(range(100)), None if adapted else probabilities)
        details[tag] = values.pop("per_class")
        records.append({"task": "cifar100", "method": method, "seed": seed, "samples": len(ids), **values})
        comparisons.append({"method": method, "seed": seed, "accuracy_delta_vs_flat": float((predicted == truth).mean() - (baseline == truth).mean()),
                            "corrected": int(((baseline != truth) & (predicted == truth)).sum()),
                            "harmed": int(((baseline == truth) & (predicted != truth)).sum())})
        for index, label, pred in zip(ids, labels, predicted):
            predictions.append({"task": "cifar100", "method": method, "seed": seed, "id": f"{binary_split}:{index}",
                                "truth": label, "predicted": int(pred)})
        print(f"{method} seed={seed}: accuracy={values['accuracy']:.4f}, macro_f1={values['macro_f1']:.4f}", flush=True)
    table(output / "comparisons.tsv", comparisons)
    routing = {
        "coarse_accuracy": float((coarse_prob.argmax(1) == parent_truth).mean()),
        "flat_induced_coarse_accuracy": float((mapping[baseline] == parent_truth).mean()),
        "hard_wrong_route_count": int((coarse_prob.argmax(1) != parent_truth).sum()),
        "note": "Any hard-route error prevents correct fine classification; smoke is operational only"}
    if rerank:
        from .cifar100_rerank import ablation_comparisons
        table(output / "ablation_comparisons.tsv", ablation_comparisons(truth, scores))
        routing = {"router": "sum flat-name probabilities by group", "top_groups": spec["top_groups"],
                   "group_top1_accuracy": float((coarse_prob.argmax(1) == parent_truth).mean()),
                   "candidate_group_recall": float((candidate_groups == parent_truth[:, None]).any(1).mean()),
                   "candidate_fine_classes": 5 * spec["top_groups"],
                   "note": "Candidate recall is not final classification accuracy; no ground-truth routing"}
    if adapted:
        from .cifar100_em_stata import majority
        routing['adapted_coarse_accuracy'] = {}
        for seed in spec['grouping_seeds']:
            votes = adaptation_arrays[f'coarse_votes_{seed}']
            routing['adapted_coarse_accuracy'][str(seed)] = {
                'em': float((votes[:, 1] == parent_truth).mean()),
                'stata': float((votes[:, 2] == parent_truth).mean()),
                'majority': float((majority(votes, 20) == parent_truth).mean())}
    write_json(output / "routing.json", routing)
    np.savez_compressed(output / "replay_scores.npz", ids=ids, labels=truth, coarse_labels=parent_truth,
                        fine_logits=fine_logits, mapping=mapping, **replay_extra,
                        **{method: values for method, values in scores.items()})
    plan.update(elapsed_seconds=time.perf_counter()-started, selected_records_sha256=record_hash.hexdigest())
    return publish(output, records, predictions, details, plan)
