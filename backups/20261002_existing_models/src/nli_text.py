"""Optional, frozen NLI text branch; no downstream fitting or label access.

Method inspiration: https://aclanthology.org/D19-1404/ (not an exact reproduction).
Single-label scoring follows Hugging Face's zero-shot pipeline: softmax of
entailment logits ACROSS candidate labels, not binary entailment probabilities.
The multimodal probability fusion remains a project-specific extension.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/bart-large-mnli"
DEFAULT_REVISION = "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
DEFAULT_HYPOTHESES = Path(__file__).resolve().parents[1] / "configs/prompts/fakeddit_6way_nli.json"


def validate_backend(task_config, dataset):
    backend = task_config.get("text_backend", "clip_text")
    if backend not in ("clip_text", "nli"):
        raise ValueError("text_backend must be clip_text or nli")
    if backend == "nli" and (dataset != "fakeddit" or task_config.get("task") != "6way"):
        raise ValueError("NLI currently supports Fakeddit 6way only; other tasks must use clip_text")
    return backend


def make_text_scorer(task_config, dataset, label_order):
    if validate_backend(task_config, dataset) == "clip_text":
        return None
    return NLITextScorer(task_config.get("nli", {}), label_order)


def backend_summary(scorer):
    if scorer is None:
        return {"text_backend": "clip_text"}
    return {"text_backend": "nli", "nli_model": scorer.settings["model"],
            "nli_status": scorer.runtime["status"]}


def backend_manifest(scorer):
    return {**backend_summary(scorer), **({"nli": scorer.manifest()} if scorer is not None else {})}


class NLITextScorer:
    """Load once per inference call; release before the next dataset/model.

    Batch size counts premise/hypothesis PAIRS, not original samples. Empty
    texts are never sent to NLI. Uniform placeholder rows are excluded by the
    caller's existing image fallback and text-branch diagnostic mask.
    """
    def __init__(self, settings, label_order):
        if not isinstance(settings, dict):
            raise ValueError("nli must be a config mapping")
        allowed = {"model", "revision", "device", "batch_size", "max_length", "hypotheses_path"}
        if set(settings) - allowed:
            raise ValueError(f"Unknown NLI settings: {sorted(set(settings) - allowed)}")
        self.settings = {"model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
                         "device": "cpu", "batch_size": 4, "max_length": 1024,
                         "hypotheses_path": str(DEFAULT_HYPOTHESES), **settings}
        for key in ("batch_size", "max_length"):
            value = self.settings[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"nli.{key} must be a positive integer")
        for key in ("model", "revision", "device", "hypotheses_path"):
            if not isinstance(self.settings[key], str) or not self.settings[key].strip():
                raise ValueError(f"nli.{key} must be a nonempty string")
        if self.settings["model"] != DEFAULT_MODEL and "revision" not in settings:
            raise ValueError("Set nli.revision explicitly when changing the NLI checkpoint")
        self.labels = list(label_order)
        if len(self.labels) < 2 or len(set(map(str, self.labels))) != len(self.labels):
            raise ValueError("NLI requires at least two unique candidate labels")
        path = Path(self.settings["hypotheses_path"])
        raw = path.read_bytes()
        spec = json.loads(raw)
        hypotheses = spec["hypotheses"]
        if set(hypotheses) != {str(label) for label in self.labels}:
            raise ValueError("NLI hypothesis IDs must exactly match prompt label IDs")
        self.hypotheses = [hypotheses[str(label)] for label in self.labels]
        if any(not isinstance(h, str) or not h.strip() for h in self.hypotheses):
            raise ValueError("NLI hypotheses must be nonempty strings")
        if len(set(self.hypotheses)) != len(self.hypotheses):
            raise ValueError("NLI hypotheses must be distinct")
        self.spec = spec
        self.hypothesis_sha256 = hashlib.sha256(raw).hexdigest()
        self.runtime = {"status": "not_run"}
        self.length_info = None

    def manifest(self):
        return {"settings": dict(self.settings), "hypothesis_spec": self.spec,
                "hypothesis_sha256": self.hypothesis_sha256,
                "label_order": self.labels, "runtime": dict(self.runtime),
                "score_rule": "softmax_across_labels_of_entailment_logits",
                "vlm_text_scale_applied": False, "vlm_bias_applied": False,
                "protocol": "exploratory_validation_no_downstream_fit"}

    def _load(self):
        # No transformers import, checkpoint download or device allocation on baseline.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = self.settings["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.settings["model"],
                                                  revision=self.settings["revision"], trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.settings["model"], revision=self.settings["revision"],
            trust_remote_code=False, use_safetensors=True)
        try:
            model.to(device)
            model.eval()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False  # no decoder KV cache needed for classification
        except Exception:
            del model
            self._release(device)
            raise
        self.runtime.update(device=str(device), revision_resolved=getattr(model.config, "_commit_hash", None))
        return tokenizer, model, device

    @staticmethod
    def _release(device):
        import gc
        import torch
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif str(device) == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def predict(self, texts, show_progress=True):
        from inference import _clean_texts, _progress_iter, _softmax
        import torch
        self.runtime = {"status": "not_run"}
        self.length_info = None
        texts = _clean_texts(texts)
        active = [i for i, t in enumerate(texts) if t]
        scores = np.full((len(texts), len(self.labels)), 1.0 / len(self.labels))
        if not active:
            self.runtime["status"] = "skipped_empty_text"
            return scores
        tokenizer = model = None
        device = "cpu"
        try:
            tokenizer, model, device = self._load()
            label2id = getattr(model.config, "label2id", {})
            entailment = [int(i) for name, i in label2id.items() if name.lower().startswith("entail")]
            if len(entailment) != 1 or not 0 <= entailment[0] < model.config.num_labels:
                raise ValueError("NLI config must explicitly identify a unique entailment label")
            context = self.settings["max_length"]
            limits = [getattr(tokenizer, "model_max_length", None),
                      getattr(model.config, "max_position_embeddings", None)]
            if any(isinstance(n, int) and 0 < n < context for n in limits):
                raise ValueError("nli.max_length exceeds tokenizer/model context")
            specials = tokenizer.num_special_tokens_to_add(pair=True)
            hypothesis_lengths = [len(tokenizer.encode(h, add_special_tokens=False, truncation=False))
                                  for h in self.hypotheses]
            if max(hypothesis_lengths) + specials >= context:
                raise ValueError("NLI hypothesis too long: no room for premise")
            # Maximum pair length across labels for each sample, including special tokens.
            lengths = [None] * len(texts)
            for i in active:
                lengths[i] = len(tokenizer.encode(texts[i], add_special_tokens=False, truncation=False)) + max(hypothesis_lengths) + specials
            self.length_info = {"lengths": lengths, "context": context, "status": "measured",
                                "unit": "max_premise_hypothesis_pair_tokens"}
            n_truncated = sum(lengths[i] > context for i in active)
            self.runtime.update(status="running", entailment_id=entailment[0],
                                n_texts=len(active), n_pairs=len(active) * len(self.labels),
                                n_truncated=n_truncated, truncated_rate=n_truncated / len(active))
            if n_truncated:
                print(f"[NLI LENGTH] {n_truncated}/{len(active)} texts truncated for at least one hypothesis; only premise is truncated")
            logits = np.empty((len(active), len(self.labels)), dtype=np.float32)
            batch_size = self.settings["batch_size"]
            iterator = _progress_iter(range(0, logits.size, batch_size), show_progress,
                                      "NLI text inference", "batch")
            for start in iterator:
                pairs = [divmod(j, len(self.labels)) for j in range(start, min(start + batch_size, logits.size))]
                encoded = tokenizer([texts[active[i]] for i, _ in pairs],
                                    [self.hypotheses[j] for _, j in pairs],
                                    padding=True, truncation="only_first", max_length=context,
                                    return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    output = model(**encoded)
                values = output.logits.detach().float().cpu().numpy()
                if values.shape != (len(pairs), model.config.num_labels) or not np.isfinite(values).all():
                    raise ValueError("Invalid NLI logits")
                for (i, j), value in zip(pairs, values[:, entailment[0]]):
                    logits[i, j] = value
                del output, encoded
            scores[active] = _softmax(logits, axis=-1)
            self.runtime["status"] = "applied"
            return scores
        finally:
            del model, tokenizer
            self._release(device)
