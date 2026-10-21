"""Đối chiếu báo cáo từ kết quả đã lưu; không đọc raw/processed, không chạy model."""
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             hamming_loss)
import yaml

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {"fakeddit": "455fc8feb8733f1d394c", "crisismmd": "0451b45a4274038830ed",
           "mmimdb": "c308fdf53c1f0b1ff369"}


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def multi_labels(value):
    if value.startswith("["):
        return sorted(ast.literal_eval(value))
    return sorted(filter(None, value.split("|")))


def binary_matrix(values, labels):
    return np.array([[int(label in row) for label in labels] for row in values])


def metrics(truth, predicted, labels, multi=False):
    if multi:
        truth, predicted = binary_matrix(truth, labels), binary_matrix(predicted, labels)
        return {"micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
                "samples_f1": float(f1_score(truth, predicted, average="samples", zero_division=0)),
                "hamming_accuracy": float(1-hamming_loss(truth, predicted)),
                "exact_match": float(accuracy_score(truth, predicted))}
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(truth, predicted, labels=labels, average="micro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted))}


def analyze():
    profile = yaml.safe_load((ROOT / "configs/launcher_3000.yaml").read_text())
    wanted = {(job["dataset"], job["task"], model, "adapted_dclip" if job["kind"] == "adapted" else "baseline")
              for job in profile["jobs"].values() for model in job["models"]}
    accepted, excluded, arrays = {}, [], {}
    for path in sorted((ROOT / "results").rglob("summary.tsv")):
        relative = path.relative_to(ROOT)
        if "samples_3000" not in relative.parts or "_system" in relative.parts:
            continue
        receipt_file = path.parent / "run_receipt.json"
        receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {}
        cohort = receipt.get("cohort") or {}
        if cohort.get("cohort_id") not in COHORTS.values():
            excluded.append({"summary": str(relative), "reason": "Không có biên nhận cohort chính hiện tại"})
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            summaries = list(csv.DictReader(stream, delimiter="\t"))
        assert len(summaries) == 1, relative
        summary = summaries[0]
        dataset, task, model = (summary[k] for k in ("dataset", "task", "model"))
        method = relative.parts[1]
        key = (dataset, task, model, method)
        assert key in wanted and key not in accepted, key
        assert cohort["cohort_id"] == COHORTS[dataset] and cohort["task"] == task
        assert cohort["cohort_samples"] == cohort["requested_samples"] == int(summary["n_samples"]) == 3000
        assert receipt["run_id"] == path.parent.name.rsplit("_", 1)[-1]
        with zipfile.ZipFile(path.parent / "run_artifacts.zip") as archive:
            def member(ending):
                found = [name for name in archive.namelist() if name.endswith(ending)]
                assert len(found) == 1, (relative, ending, found)
                return found[0]
            manifest = json.loads(archive.read(member("/manifest.json")))
            snapshot = json.loads(archive.read(member("/evaluation_snapshot.json")))
            with archive.open(member("/raw_predictions.tsv")) as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"), delimiter="\t")
                fields = reader.fieldnames
                labels = [field[len("score_image_"):] for field in fields if field.startswith("score_image_")]
                multi = dataset == "mmimdb"
                prefix = "zscore_" if multi else "score_"
                rows = [{name: row[name] for name in fields if name not in ("text", "image_path")}
                        for row in reader]
        assert len(rows) == manifest["n_total_samples"] == 3000
        assert len({row["id"] for row in rows}) == 3000
        assert manifest["model"] == model and manifest["dataset"] == dataset
        assert not manifest["relation_enabled"]
        truth = [multi_labels(row["y_true"]) if multi else row["y_true"] for row in rows]
        predicted = [multi_labels(row["y_pred"]) if multi else row["y_pred"] for row in rows]
        score = np.array([[float(row[prefix+label]) for label in labels] for row in rows])
        image = np.array([[float(row["score_image_"+label]) for label in labels] for row in rows])
        text = np.array([[float(row["score_text_"+label]) for label in labels] for row in rows])
        assert np.isfinite(score).all() and np.isfinite(image).all() and np.isfinite(text).all()
        def decode(values):
            if not multi:
                return [labels[index] for index in values.argmax(axis=1)]
            result = []
            for row in values:
                chosen = [label for label, value in zip(labels, row) if value > manifest["relative_z_threshold"]]
                if not chosen and manifest["fallback_top1"]:
                    chosen = [labels[int(row.argmax())]]
                result.append(sorted(chosen))
            return result
        assert decode(score) == predicted, relative
        measured = metrics(truth, predicted, labels, multi)
        for name, value in measured.items():
            if summary.get(name):
                assert abs(float(summary[name])-value) < 1e-10, (relative, name, value, summary[name])
        branch = {name: metrics(truth, decode(values), labels, multi) for name, values in (("image",image),("text",text))}
        for name in branch:
            assert abs(branch[name]["macro_f1"]-float(summary[name+"_macro_f1"])) < 1e-10
        by_id = {sample["id"]: sample for sample in snapshot["cohort"]["samples"]}
        assert set(by_id) == {row["id"] for row in rows}
        content = []
        for row, target in zip(rows, truth):
            item = by_id[row["id"]]
            saved_truth = multi_labels(item["label"]) if multi else item["label"]
            assert saved_truth == target
            content.append((row["id"], item["text_sha256"], target,
                            Path(item["image_path"]).name, item["image_stat"]["size"]))
        report_truth = binary_matrix(truth, labels) if multi else truth
        report_pred = binary_matrix(predicted, labels) if multi else predicted
        details = classification_report(report_truth, report_pred,
            labels=list(range(len(labels))) if multi else labels,
            target_names=labels, output_dict=True, zero_division=0)
        counts = Counter(label for targets in truth for label in targets) if multi else Counter(truth)
        record = {"dataset":dataset,"task":task,"model":model,"method":method,
                  "run_id":receipt["run_id"],"summary_path":str(relative),"receipt":receipt,
                  "n_samples":len(rows),"labels":labels,"support":dict(counts),
                  "metrics":measured,"branch_metrics":branch,"per_class":{label:details[label] for label in labels},
                  "cohort_content_sha256":signature(sorted(content)),
                  "shared_input_sha256":signature(sorted((item[0],item[1],item[3],item[4]) for item in content)),
                  "config":{k:manifest.get(k) for k in ("image_weight","text_logit_scale_effective","prompt_aggregation",
                    "image_prompt_aggregation","descriptor_sha256","descriptor_provenance","decision_mode",
                    "relative_z_threshold","fallback_top1","chunk_long_text","chunk_aggregation","batch_size","logit_scale","logit_bias")},
                  "truncation":snapshot["truncation"],"chunking":snapshot.get("chunking"),
                  "empty_texts":sum(row.get("_text_was_empty")=="True" for row in rows)}
        if not multi:
            record["confusion_matrix"] = confusion_matrix(truth,predicted,labels=labels).tolist()
            majority = max(counts,key=counts.get)
            record["majority_baseline"] = metrics(truth,[majority]*len(truth),labels)
        else:
            record["mean_true_labels"] = float(np.mean([len(t) for t in truth]))
            record["mean_predicted_labels"] = float(np.mean([len(t) for t in predicted]))
            record["top1_fallback_samples"] = sum(row.get("_used_top1_fallback")=="True" for row in rows)
        accepted[key] = record
        arrays[key] = {"ids":[row["id"] for row in rows],"truth":truth,"pred":predicted,"text":text}
    assert set(accepted) == wanted, ("missing",wanted-set(accepted))
    for dataset in COHORTS:
        group = [value for key,value in accepted.items() if key[0] == dataset]
        assert len({value["shared_input_sha256"] for value in group}) == 1, (dataset,"different inputs across tasks")
    for dataset, task in {(key[0],key[1]) for key in accepted}:
        group = [value for key,value in accepted.items() if key[:2]==(dataset,task)]
        assert len({value["cohort_content_sha256"] for value in group}) == 1, (dataset,task,"different content")
        assert len({value["receipt"]["data_sha256"] for value in group}) == 1, (dataset,task,"different data bundle")
    pairs = []
    for key, adapted in accepted.items():
        if key[3] != "adapted_dclip": continue
        base_key = (*key[:3],"baseline")
        base = accepted[base_key]
        a,b = arrays[base_key],arrays[key]
        assert a["ids"] == b["ids"] and a["truth"] == b["truth"]
        max_text_difference = float(np.max(np.abs(a["text"]-b["text"])))
        assert max_text_difference < 1e-5, (key,"text branch changed",max_text_difference)
        good_a = np.array([p==t for p,t in zip(a["pred"],a["truth"])])
        good_b = np.array([p==t for p,t in zip(b["pred"],b["truth"])])
        pairs.append({"task":key[1],"model":key[2],"baseline_run":base["run_id"],"dclip_run":adapted["run_id"],
                      "delta_accuracy":adapted["metrics"]["accuracy"]-base["metrics"]["accuracy"],
                      "delta_macro_f1":adapted["metrics"]["macro_f1"]-base["metrics"]["macro_f1"],
                      "corrected":int(np.sum(~good_a & good_b)),"introduced_errors":int(np.sum(good_a & ~good_b)),
                      "text_branch_max_difference":max_text_difference})
    return {"scope":"28 lượt cohort chính, 3000 mẫu/lượt; không chạy lại model",
            "checked_runs":len(accepted),"prediction_rows_checked":sum(x["n_samples"] for x in accepted.values()),
            "excluded_runs":excluded,"runs":list(accepted.values()),"paired_comparisons":pairs,
            "checks":{"matrix_complete":True,"metrics_recomputed":True,"predictions_match_scores":True,
                      "cohort_content_equal_per_task":True,"inputs_equal_across_tasks_per_dataset":True,
                      "paired_text_scores_equal_within_1e_5":True}}


def main():
    result = analyze()
    folder = ROOT / "docs/experiments"
    (folder/"ket_qua_3000_kiem_tra.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    with (folder/"ket_qua_3000.tsv").open("w",newline="",encoding="utf-8") as stream:
        fields = ["dataset","task","model","method","run_id","n_samples","accuracy","macro_f1",
                  "micro_f1","balanced_accuracy","samples_f1","hamming_accuracy","exact_match"]
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter="\t")
        writer.writeheader()
        for run in result["runs"]:
            writer.writerow({**{name:run[name] for name in fields[:6]},**run["metrics"]})
    print(json.dumps({key:result[key] for key in ("checked_runs","prediction_rows_checked","excluded_runs","checks","paired_comparisons")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
