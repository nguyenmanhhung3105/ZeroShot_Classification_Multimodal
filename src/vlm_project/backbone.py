"""Một backbone duy nhất: MetaCLIP2 ViT-H/14, không huấn luyện hoặc fine-tune."""
import numpy as np
from .io import MODEL, digest
ARCH = 'ViT-H-14-worldwide-quickgelu'
PRETRAINED = 'metaclip2_worldwide'
TEXT_BATCH_SIZE = 64

class Backbone:

    def __init__(self, device='auto'):
        import torch
        import open_clip
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        if device == 'cuda' and (not torch.cuda.is_available()):
            raise ValueError('CUDA chưa bật; trên Kaggle chọn GPU T4 x2')
        self.device = device
        self.name = MODEL
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
        self.model.to(device).eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(ARCH)
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self._prototype_cache = {}
        self._text_batch_size = TEXT_BATCH_SIZE
        self.metadata = {
            'architecture': ARCH,
            'pretrained': PRETRAINED,
            'tokenizer': 'facebook/xlm-v-base',
            'context_length': self.model.context_length,
            'device': device,
            'dtype': str(next(self.model.parameters()).dtype),
            'image_size': self.model.visual.image_size,
            'logit_scale_checkpoint': self.logit_scale,
            'text_batch_size': TEXT_BATCH_SIZE,
            'text_batch_size_effective': TEXT_BATCH_SIZE,
            'text_batching': 'across classes, ordered class means; no autocast',
            'prototype_cache': 'in-memory, exact ordered prompt bank, same loaded model',
        }
        print(f'Model {MODEL}; device={device}; context={self.model.context_length}', flush=True)

    def check_prompts(self, texts):
        if not hasattr(self.tokenizer, 'tokenizer') or not hasattr(self.tokenizer, 'clean_fn'):
            raise ValueError('Cần HFTokenizer của MetaCLIP2 worldwide')
        encoded = self.tokenizer.tokenizer([self.tokenizer.clean_fn(t) for t in texts], add_special_tokens=True, truncation=False, padding=False)['input_ids']
        if any((len(tokens) > self.model.context_length for tokens in encoded)):
            raise ValueError('Prompt vượt context; không cắt ngầm prompt nhãn')

    def encode_images(self, images):
        with self.torch.no_grad():
            batch = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
            x = self.model.encode_image(batch)
            return (x / x.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def encode_texts(self, texts, strict=False):
        if strict:
            self.check_prompts(texts)
        with self.torch.no_grad():
            tokens = self.tokenizer(texts, context_length=self.model.context_length).to(self.device)
            x = self.model.encode_text(tokens)
            return (x / x.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def prototypes(self, bank):
        """Mean cosine: trung bình embedding unit, không chuẩn hóa lại trung bình."""
        if not bank or any(not texts for texts in bank.values()):
            raise ValueError('Prompt bank phải có ít nhất một câu mỗi lớp')
        # List giữ thứ tự lớp trong cache key; digest(dict) tự sắp khóa sẽ sai.
        key = digest(list(bank.items()))
        if key in self._prototype_cache:
            print(f'Prompt cache: dùng lại {len(bank)} lớp', flush=True)
            return self._prototype_cache[key].copy()
        texts = [text for values in bank.values() for text in values]
        counts = [len(values) for values in bank.values()]
        print(f'Mã hóa prompt: {len(bank)} lớp / {len(texts)} câu, batch={TEXT_BATCH_SIZE}', flush=True)
        blocks = []
        start = 0
        while start < len(texts):
            batch_size = self._text_batch_size
            try:
                block = self.encode_texts(texts[start:start+batch_size], strict=True)
            except self.torch.cuda.OutOfMemoryError:
                if self.device != 'cuda' or batch_size <= 16:
                    raise
                self._text_batch_size = max(16,batch_size//2)
                self.metadata['text_batch_size_effective'] = self._text_batch_size
                self.torch.cuda.empty_cache()
                print(f'VRAM thiếu cho prompt batch={batch_size}; thử lại batch={self._text_batch_size}', flush=True)
                continue
            blocks.append(block)
            done = min(start+batch_size,len(texts))
            if start == 0 or done % 1024 == 0 or done == len(texts):
                print(f'Mã hóa prompt: {done}/{len(texts)}', flush=True)
            start = done
        values = np.concatenate(blocks)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        offsets = np.cumsum([0,*counts])
        vectors = np.asarray([values[a:b].mean(0) for a,b in zip(offsets[:-1],offsets[1:])])
        self._prototype_cache[key] = vectors
        return vectors.copy()

    def unload(self):
        self._prototype_cache.clear()
        del self.model
        if self.device == 'cuda':
            self.torch.cuda.empty_cache()
        if self.device == 'mps':
            self.torch.mps.empty_cache()

def encode_cohort(backend, rows, batch_size, with_titles=False):
    """Mỗi ảnh mở đúng batch; ảnh lỗi làm dừng thay vì âm thầm đổi cohort."""
    from PIL import Image
    images_out = []
    titles = []
    for start in range(0, len(rows), batch_size):
        images = []
        try:
            for row in rows[start:start + batch_size]:
                with Image.open(row['image']) as image:
                    images.append(image.convert('RGB'))
            images_out.append(backend.encode_images(images))
            if with_titles:
                titles.append(backend.encode_texts([r['text'] for r in rows[start:start + batch_size]]))
        finally:
            for image in images:
                image.close()
        done = min(start + batch_size, len(rows))
        if start == 0 or done % 128 == 0 or done == len(rows):
            print(f'Mã hóa ảnh: {done}/{len(rows)}', flush=True)
    return (np.concatenate(images_out), np.concatenate(titles) if with_titles else None)
