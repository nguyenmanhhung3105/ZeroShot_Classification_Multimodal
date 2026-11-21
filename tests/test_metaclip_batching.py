"""Vector tổng hợp: giữ mean cosine, kiểm tra cache và smoke/full một model."""
import contextlib
import hashlib
import io as stdio
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_project import backbone, benchmark, io, packaging, protocol


def vectors(texts):
    raw=np.asarray([list(hashlib.sha256(text.encode()).digest()[:8]) for text in texts],dtype=np.float32)
    return raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)


class SyntheticBackend(backbone.Backbone):
    def __init__(self,device='cpu'):
        self.device=device
        self._prototype_cache={}
        self._text_batch_size=backbone.TEXT_BATCH_SIZE
        self.metadata={'synthetic':True,'dtype':'float32','text_batch_size_effective':64}
        self.encoded=[]
        self.unloads=0

    def encode_texts(self,texts,strict=False):
        assert strict
        self.encoded.append(list(texts))
        return vectors(texts)

    def unload(self):
        self.unloads+=1
        self._prototype_cache.clear()


class MetaCLIPBatchingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_cross_class_batches_preserve_old_normalization_and_ordered_means(self):
        bank={f'class{i}':[f'prompt-{i}-{j}' for j in range(size)]
              for i,size in enumerate([20,40,3,2])}
        old=[]
        for texts in bank.values():
            values=np.concatenate([vectors(texts[i:i+16]) for i in range(0,len(texts),16)])
            values/=np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)
            old.append(values.mean(0))
        model=SyntheticBackend()
        with contextlib.redirect_stdout(stdio.StringIO()):actual=model.prototypes(bank)
        np.testing.assert_allclose(actual,old,rtol=1e-6,atol=1e-7)
        self.assertEqual(actual.dtype,np.float32)
        self.assertEqual([len(batch) for batch in model.encoded],[64,1])
        images=vectors(['query1','query2','query3'])
        np.testing.assert_array_equal((images@actual.T).argmax(1),(images@np.asarray(old).T).argmax(1))

    def test_cache_is_prompt_and_class_order_sensitive_and_returns_copies(self):
        model=SyntheticBackend();bank={'a':['one','two'],'b':['three']}
        with contextlib.redirect_stdout(stdio.StringIO()):
            original=model.prototypes(bank);first=original.copy();original[:]=0
            cached=model.prototypes(bank)
            self.assertEqual(len(model.encoded),1)
            np.testing.assert_array_equal(cached,first)
            swapped=model.prototypes(dict(reversed(list(bank.items()))))
            np.testing.assert_allclose(swapped,first[::-1])
            self.assertEqual(len(model.encoded),2)
            changed=model.prototypes({'a':['different'],'b':['three']})
            self.assertEqual(len(model.encoded),3)
            self.assertFalse(np.array_equal(changed[0],first[0]))

    def test_cuda_oom_retries_same_texts_with_smaller_batch_without_dropping_any(self):
        class FakeOOM(RuntimeError):pass
        model=SyntheticBackend('cuda')
        empty=SimpleNamespace(calls=0)
        def clear():empty.calls+=1
        model.torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=FakeOOM,empty_cache=clear))
        successful=[]
        def encode(texts,strict=False):
            if len(texts)>32:raise FakeOOM('synthetic OOM')
            successful.extend(texts)
            return vectors(texts)
        model.encode_texts=encode
        bank={'a':[f'text-{i}' for i in range(65)]}
        with contextlib.redirect_stdout(stdio.StringIO()):actual=model.prototypes(bank)
        values=vectors(bank['a']);values/=np.linalg.norm(values,axis=1,keepdims=True)
        np.testing.assert_allclose(actual[0],values.mean(0),rtol=1e-6)
        self.assertEqual(successful,bank['a'])
        self.assertEqual(empty.calls,1)
        self.assertEqual(model.metadata['text_batch_size_effective'],32)

    def test_smoke_generates_clip_only_but_full_keeps_all_waffle_seeds(self):
        calls=[]
        def generate(api,method,names,descriptor,pairs,seed):
            calls.append((method,seed))
            return {name:[method+name] for name in names},None
        api=SimpleNamespace(clip=SimpleNamespace(tokenize=lambda values:None))
        with patch.object(benchmark,'generate_prompts',side_effect=generate),contextlib.redirect_stdout(stdio.StringIO()):
            smoke=benchmark.prompt_trials(api,['class1','class2'],'unused',io.config(),smoke=True)
            self.assertEqual(calls,[('clip',0)])
            full=benchmark.prompt_trials(api,['class1','class2'],'unused',io.config())
        self.assertEqual(len(smoke),1)
        self.assertEqual(list(smoke[0]['prompts']),['class1','class2'])
        self.assertEqual(len(full),9)
        self.assertEqual([t['seed'] for t in full if t['method']=='waffle'],list(range(7)))

    def test_notebook_reuses_one_model_and_clip_cache_then_unloads_after_full(self):
        settings=io.config();settings['device']='cpu';settings['output_root']=str(self.root/'results')
        names=[f'class{i}' for i in range(47)]
        rows=[{'id':str(i),'image':'synthetic','label':i} for i in range(6)]
        io.write_json(self.root/'third_party/WaffleCLIP/descriptors/descriptors_dtd.json',
                      {name:['detail'] for name in names})
        def trials(api,names,descriptor,settings,smoke=False):
            result=[{'method':'clip','seed':None,'prompts':{n:['clip '+n] for n in names}}]
            if not smoke:
                result += [{'method':'dclip','seed':None,'prompts':{n:['dclip '+n] for n in names}}]
                result += [{'method':'waffle','seed':seed,'prompts':{n:[f'waffle {seed} '+n] for n in names}}
                           for seed in range(7)]
            return result
        def dataset(root,name,smoke):return names,rows[:5] if smoke else rows,{}
        def encode(model,rows,batch):return vectors([r['id'] for r in rows]),None
        model=SyntheticBackend()
        namespace={'TASK':'benchmark','DATASET':'dtd','SPLIT':'dev','DATA_ROOT':'synthetic',
                   'PROJECT':self.root,'settings':settings,'locate_data':lambda *args:self.root}
        document=packaging.notebook()
        with patch.object(backbone,'Backbone',return_value=model) as create,\
             patch.object(benchmark,'ROOT',self.root),patch.object(benchmark,'verify_sources',return_value={}),\
             patch.object(benchmark.data,'benchmark',side_effect=dataset),\
             patch.object(benchmark,'prompt_trials',side_effect=trials),\
             patch.object(benchmark,'author_api',return_value=None),\
             patch.object(benchmark,'encode_cohort',side_effect=encode),\
             contextlib.redirect_stdout(stdio.StringIO()):
            exec(''.join(document['cells'][3]['source']),namespace)
            self.assertEqual(model.unloads,0)
            smoke=protocol.completed_run(namespace['smoke_output'])
            self.assertEqual([r['method'] for r in smoke['summary']],['clip'])
            exec(''.join(document['cells'][4]['source']),namespace)
        self.assertEqual(create.call_count,1)
        self.assertEqual(model.unloads,1)
        full=protocol.completed_run(namespace['output'])
        self.assertEqual({r['method']:r['seeds'] for r in full['summary']},
                         {'clip':1,'dclip':1,'waffle':7,'waffle_unit_ablation':7})
        self.assertEqual(sum(text.startswith('clip ') for batch in model.encoded for text in batch),47)

    def test_notebook_unloads_shared_model_when_smoke_fails(self):
        model=SyntheticBackend();document=packaging.notebook()
        namespace={'TASK':'benchmark','DATASET':'dtd','SPLIT':'dev','DATA_ROOT':'synthetic',
                   'settings':{'device':'cpu'},'locate_data':lambda *args:self.root}
        with patch.object(backbone,'Backbone',return_value=model),\
             patch.object(benchmark,'run',side_effect=ValueError('synthetic failure')):
            with self.assertRaisesRegex(ValueError,'synthetic failure'),contextlib.redirect_stdout(stdio.StringIO()):
                exec(''.join(document['cells'][3]['source']),namespace)
        self.assertEqual(model.unloads,1)


if __name__=='__main__':unittest.main()
