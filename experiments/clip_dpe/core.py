"""DPE trên feature single-view: residual ảnh/text, cache và online averaging.

Giữ operator của main_dpe.py; CPU FP32 thay CUDA FP16. Không nhận nhãn thật.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from experiments.clip_dmn.core import inputs

REFERENCE_COMMIT = 'fecae5a50d7dc39fb944221be390201a56948416'
PARAMETERS = {'memory_size': 3, 'alpha': 6., 'beta': 5., 'learning_rate': .0006,
              'alignment_weight': .5, 'logit_scale': 100., 'text_update_entropy': .1}


def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def mean_entropy(logits):
    logp = logits - logits.logsumexp(dim=-1, keepdim=True)
    average = logp.logsumexp(dim=0) - np.log(logp.shape[0])
    average = torch.clamp(average, min=torch.finfo(average.dtype).min)
    return -(average * torch.exp(average)).sum(dim=-1)


def alignment(keys, text):
    logits = F.normalize(keys, dim=1) @ F.normalize(text, dim=1).T / .01
    return F.cross_entropy(logits, torch.arange(len(keys), device=keys.device))


def predict_stream(features, prototypes, order=None, memory_size=3, alpha=6., beta=5.,
                   learning_rate=.0006, alignment_weight=.5, logit_scale=100., text_update_entropy=.1):
    if PARAMETERS != {'memory_size': memory_size, 'alpha': alpha, 'beta': beta,
                      'learning_rate': learning_rate, 'alignment_weight': alignment_weight,
                      'logit_scale': logit_scale, 'text_update_entropy': text_update_entropy}:
        raise ValueError('Khóa một cấu hình ImageNet tác giả, không search tham số')
    x, text, order = inputs(features, prototypes, order)
    if x.shape[1] != 1:
        raise ValueError('Runner này chỉ nhận đúng một view thật mỗi ảnh')
    x = x[:, 0]
    n, dimension = x.shape; classes = len(text)
    previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with torch.enable_grad():
            global_text = text.T.contiguous(); cache = {}; updates = 0
            out = np.empty((n, classes), dtype=np.float32)
            initial = np.empty_like(out)
            trace = {name: np.empty(n, dtype=np.float32) for name in
                     ['cache_entropy', 'training_entropy', 'alignment_loss', 'post_entropy']}
            trace.update(pseudo_label=np.empty(n, dtype=np.int64), memory_write=np.zeros(n, dtype=bool),
                         replaced_index=np.full(n, -1, dtype=np.int64), text_update=np.zeros(n, dtype=bool))
            for index in order:
                local_text = global_text.clone().detach()
                residual_text = torch.nn.Parameter(torch.zeros_like(local_text))
                start_text = F.normalize(local_text + residual_text, dim=0)
                query = F.normalize(x[index:index+1], dim=-1)
                initial_logits = logit_scale * query @ start_text
                loss0 = entropy(initial_logits)
                label = int(initial_logits.topk(1, 1, True, True)[1].t()[0])
                quality = float(loss0.detach() / math.log2(classes))
                initial[index] = initial_logits.detach().numpy()[0]
                trace['pseudo_label'][index] = label; trace['cache_entropy'][index] = quality
                item = [query.detach(), quality, int(index)]
                if label not in cache:
                    cache[label] = [item]; trace['memory_write'][index] = True
                else:
                    if len(cache[label]) < memory_size:
                        cache[label].append(item); trace['memory_write'][index] = True
                    elif quality < cache[label][-1][1]:
                        trace['replaced_index'][index] = cache[label][-1][2]
                        cache[label][-1] = item; trace['memory_write'][index] = True
                    cache[label] = sorted(cache[label], key=lambda item: item[1])
                active = sorted(cache)
                # Mean cache feature chưa unit-normalize, như tác giả.
                keys = []
                for c in active:
                    key = torch.zeros_like(query)
                    for item in cache[c]:
                        key += item[0] / len(cache[c])
                    keys.append(key)
                keys = torch.cat(keys, dim=0).T
                values = F.one_hot(torch.tensor(active), num_classes=classes).float()
                residual_image = torch.nn.Parameter(torch.zeros_like(keys))
                new_text = F.normalize(local_text + residual_text, dim=0)
                new_keys = F.normalize(keys + residual_image, dim=0)
                affinity = query @ new_keys
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ values
                cache_logits = alpha * cache_logits
                training_logits = initial_logits.clone() + cache_logits
                ent = mean_entropy(training_logits)
                align = alignment(new_keys.T, new_text[:, active].T)
                loss = ent + alignment_weight * align
                optimizer = torch.optim.AdamW([
                    {'params': [residual_text], 'lr': learning_rate, 'eps': 1e-3, 'weight_decay': .1},
                    {'params': [residual_image], 'lr': learning_rate, 'eps': 1e-3, 'weight_decay': .1}])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                with torch.no_grad():
                    new_text = F.normalize(local_text + residual_text, dim=0)
                    new_keys = F.normalize(keys + residual_image, dim=0)
                    # get_clip_logits mã hóa/normalize lại view gốc khi inference.
                    actual_query = query
                    logits = logit_scale * actual_query @ new_text
                    cache_logits = ((-1) * (beta - beta * (actual_query @ new_keys))).exp() @ values
                    logits += alpha * cache_logits
                    out[index] = logits.numpy()[0]
                    post = mean_entropy(logits)
                    accepted = float(post / math.log2(classes)) < text_update_entropy
                    if accepted:
                        updates += 1
                        global_text = global_text * (updates / (updates + 1)) + new_text * (1 / (updates + 1))
                    trace['text_update'][index] = accepted
                    trace['training_entropy'][index] = float(ent.detach())
                    trace['alignment_loss'][index] = float(align.detach())
                    trace['post_entropy'][index] = float(post)
                if not np.isfinite(out[index]).all():
                    raise FloatingPointError('DPE logits không hữu hạn')
            trace['initial_logits'] = initial
            members = [[item[2] for item in cache.get(c, [])] for c in range(classes)]
            diagnostics = {'memory_indices': members, 'memory_counts': [len(v) for v in members],
                           'text_updates': updates, 'global_text': global_text.T.detach().tolist(),
                           'views': 1, 'order': order.tolist(), 'label_fit': False, 'support_samples': 0,
                           'residual_steps': 1, 'optimizer_reset': 'every image',
                           'cache_entropy_denominator': 'log2(classes), with natural-log entropy',
                           'update_before_prediction': True, 'memory_reset': 'every task/split/order',
                           'dtype': 'CPU float32', 'single_view_adaptation': True}
            return out, trace, diagnostics
    finally:
        torch.set_num_threads(previous_threads)
