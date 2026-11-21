"""Dev-only controls for the original Fakeddit real/fake -> five fake types."""
import time
import numpy as np
from scipy.special import softmax

from . import data
from .adaptation import probability_features, em_dirichlet, stata
from .backbone import Backbone, encode_cohort
from .evaluate import metric, publish, table
from .io import ROOT, MODEL, code_hashes, read_json, sha256, write_json, new_run
from .methods import hierarchy, fuse, majority as original_majority


def specification():
    value = read_json(ROOT / 'configs/fakeddit_hierarchy_dev.json')
    expected = {'version':1, 'samples':3000, 'group_size':75, 'grouping_seeds':[0,1,2],
                'image_scale':100., 'text_scale':25., 'image_weight':.75, 'em_temperature':30.,
                'em_iterations':20, 'em_mm_iterations':1000, 'stata_iterations':10, 'stata_neighbors':3}
    if value != expected:
        raise ValueError('Keep the fixed dev hierarchy protocol; no test or grid search')
    return value


def unit(values):
    values = np.asarray(values)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm <= 1e-12):
        raise ValueError('Invalid prototypes')
    return values / norm


def multiclass_vote(votes, classes):
    votes = np.asarray(votes)
    if votes.ndim != 2 or votes.shape[1] != 3 or not np.issubdtype(votes.dtype, np.integer):
        raise ValueError('Expected three integer votes')
    if np.any(votes < 0) or np.any(votes >= classes):
        raise ValueError('Vote outside taxonomy')
    predicted = votes[:, 0].copy()
    agree = votes[:, 1] == votes[:, 2]
    predicted[agree] = votes[agree, 1]
    return predicted


def predict(features, titles, binary_raw, six_raw, text_six_raw, spec,
            em_solver=em_dirichlet, stata_solver=stata, legacy_solver=original_majority):
    """No labels accepted. Controlled image-only comparisons + separate legacy MM."""
    import torch
    features, titles = np.asarray(features), np.asarray(titles)
    binary_raw,six_raw,text_six_raw=map(np.asarray,[binary_raw,six_raw,text_six_raw])
    if features.ndim != 2 or features.shape != titles.shape:
        raise ValueError('Image/title features must align')
    if not np.isfinite(features).all() or not np.isfinite(titles).all():
        raise ValueError('Nonfinite image/title features')
    for values in [features, titles]:
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
            raise ValueError('Expected unit encoded features')
    binary, six = unit(binary_raw), unit(six_raw)
    if binary.shape != (2,features.shape[1]) or six.shape != (6,features.shape[1]) or np.asarray(text_six_raw).shape != six.shape:
        raise ValueError('Expected 2way fake/real and 6way True/fake taxonomy')
    cosines = {'binary':features @ binary.T, 'six':features @ six.T}
    clip = {key:softmax(spec['image_scale']*value,axis=1) for key,value in cosines.items()}
    legacy_mm = fuse(softmax(spec['image_scale']*(features @ six_raw.T),axis=1),
                     softmax(spec['text_scale']*(titles @ text_six_raw.T),axis=1),spec['image_weight'])
    def routed(gate, scores):
        return hierarchy(np.asarray(gate).astype(str),scores,list('012345')).astype(int)
    results = [('flat_clip_image',None,clip['six'].argmax(1)),
               ('hierarchy_clip_image',None,routed(clip['binary'].argmax(1),clip['six'])),
               ('legacy_flat_mm',None,legacy_mm.argmax(1)),
               ('legacy_hierarchy_clip_mm',None,routed(clip['binary'].argmax(1),legacy_mm))]
    arrays = {'p_clip_binary':clip['binary'], 'p_clip_six':clip['six'], 'p_legacy_mm':legacy_mm}
    routing = {}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for seed in spec['grouping_seeds']:
            groups = np.empty(len(features),dtype=np.int32)
            predictions = {level:{'em':p.argmax(1).copy(),'stata':p.argmax(1).copy()} for level,p in clip.items()}
            order = np.random.default_rng(seed).permutation(len(features))
            for group,start in enumerate(range(0,len(features),spec['group_size'])):
                ids=order[start:start+spec['group_size']]
                groups[ids]=group
                for level,proto in [('binary',binary),('six',six)]:
                    query=probability_features(cosines[level][ids],spec['em_temperature'])
                    em_labels,_,_=em_solver(query[None],iterations=spec['em_iterations'],mm_iterations=spec['em_mm_iterations'])
                    predictions[level]['em'][ids]=em_labels[0]
                    if len(ids)>spec['stata_neighbors']:
                        _,posterior=stata_solver(features[ids],proto,max_iter=spec['stata_iterations'],neighbors=spec['stata_neighbors'])
                        predictions[level]['stata'][ids]=posterior.argmax(1)
                print(f'Fakeddit seed={seed}: adaptation group {group+1}/{(len(features)+74)//75}',flush=True)
            for level,classes in [('binary',2),('six',6)]:
                votes=np.stack([clip[level].argmax(1),predictions[level]['em'],predictions[level]['stata']],axis=1)
                predictions[level]['majority']=multiclass_vote(votes,classes)
                arrays[f'votes_{level}_s{seed}']=votes
            arrays[f'groups_s{seed}']=groups
            for method in ['em','stata','majority']:
                results.append((f'flat_{method}_image',seed,predictions['six'][method]))
                gate=predictions['binary'][method]
                results.append((f'hierarchy_{method}_image',seed,routed(gate,clip['six'])))
                routing[f'{method}_s{seed}']=gate
            # Call the canonical existing binary Majority on RAW semantic means:
            # its EM prior differs from the normalized controlled branches above.
            legacy_gate,detail=legacy_solver(features,binary_raw,seed)
            np.testing.assert_array_equal(detail['batch_ids'],groups)
            results.append(('original_hierarchy_mm',seed,routed(legacy_gate,legacy_mm)))
            arrays[f'legacy_votes_s{seed}']=detail['votes']
            routing[f'original_s{seed}']=legacy_gate
    finally:
        torch.set_num_threads(previous_threads)
    return results,arrays,routing


def run(settings,split,root,smoke=False,backend_factory=Backbone,calip=False):
    if calip:
        from .fakeddit_calip_ablation import run as calip_run
        return calip_run(settings,split,root,smoke,backend_factory)
    if split!='dev':
        raise ValueError('This job is dev-only; do not open or lock test')
    spec=specification()
    profile=read_json(settings['fakeddit']['prompt_file'])
    if profile['model']!=MODEL or profile['grouping_seeds']!=[0,1,2] or profile['rules']!={'image_logit_scale':100.,'text_logit_scale':25.,'image_weight':.75}:
        raise ValueError('Original prompt profile/rules changed')
    for bank,size in [(profile['image_prompts']['2way']['semantic_v1'],2),
                      (profile['image_prompts']['6way']['legacy'],6),(profile['text_prompts']['6way'],6)]:
        if list(bank)!=[str(i) for i in range(size)] or any(not texts for texts in bank.values()):
            raise ValueError('Prompt class order must match the original profile')
    rows,cohort=data.fakeddit(root,'dev',smoke)
    if len(rows)!=(5 if smoke else 3000):
        raise ValueError('Unexpected dev cohort size')
    output=new_run(settings,'fakeddit_hierarchy','dev',smoke)
    plan={'kind':'fakeddit_hierarchy','dataset':'fakeddit_6way','model':MODEL,'split':'dev',
          'mode':'smoke' if smoke else 'full','samples':len(rows),'cohort':cohort,'specification':spec,
          'code_sha256':code_hashes(),'profile_sha256':sha256(settings['fakeddit']['prompt_file']),
          'specification_sha256':sha256(ROOT/'configs/fakeddit_hierarchy_dev.json'),
          'requirements_sha256':sha256(ROOT/'requirements.txt'),
          'scope':'Matched image-only flat/hierarchy controls; original multimodal reference evaluated separately.',
          'transductive':'Unlabeled groups of 75, seeds 0/1/2; no supervised weight updates',
          'normalization':'Controlled branches use unit means; original reference retains raw mean EM and legacy MM scores',
          'oracle':'Ground-truth binary gate diagnostics only; excluded from method ranking',
          'profile':profile,'batch_size':settings['batch_size']}
    write_json(output/'frozen_plan.json',plan)
    write_json(output/'run.json',{**plan,'status':'running'})
    start=time.perf_counter()
    backend=backend_factory(settings['device'])
    try:
        binary=backend.prototypes(profile['image_prompts']['2way']['semantic_v1'])
        six=backend.prototypes(profile['image_prompts']['6way']['legacy'])
        text_six=backend.prototypes(profile['text_prompts']['6way'])
        features,titles=encode_cohort(backend,rows,settings['batch_size'],with_titles=True)
        plan['model_runtime']=getattr(backend,'metadata',{'backend':type(backend).__name__})
    finally:
        backend.unload()
    results,arrays,gates=predict(features,titles,binary,six,text_six,spec)
    # Ground truth is introduced only after all real-method predictions exist.
    truth=np.asarray([int(row['6way']) for row in rows])
    binary_truth=np.asarray([int(row['2way']) for row in rows])
    if not np.array_equal(binary_truth,(truth==0).astype(int)):
        raise ValueError('Fakeddit binary/six-way taxonomy mismatch')
    records,predictions,details,comparisons=[],[],{},[]
    image_reference=results[0][2];mm_reference=results[2][2]
    for method,seed,predicted in results:
        values=metric(truth,predicted,list(range(6)))
        key=method if seed is None else f'{method}_s{seed}'
        details[key]=values.pop('per_class')
        records.append({'task':'6way','method':method,'seed':seed,'samples':len(rows),**values})
        reference=mm_reference if method in ['legacy_flat_mm','legacy_hierarchy_clip_mm','original_hierarchy_mm'] else image_reference
        reference_name='legacy_flat_mm' if method in ['legacy_flat_mm','legacy_hierarchy_clip_mm','original_hierarchy_mm'] else 'flat_clip_image'
        comparisons.append({'method':method,'seed':seed,'reference':reference_name,
            'accuracy_delta':float((predicted==truth).mean()-(reference==truth).mean()),
            'corrected':int(((reference!=truth)&(predicted==truth)).sum()),
            'harmed':int(((reference==truth)&(predicted!=truth)).sum())})
        predictions.extend({'task':'6way','method':method,'seed':seed,'id':row['id'],'truth':int(y),'predicted':int(p)}
                           for row,y,p in zip(rows,truth,predicted))
        arrays[f'prediction_{key}']=predicted
        print(f'{method} seed={seed}: accuracy={values["accuracy"]:.4f}, macro_f1={values["macro_f1"]:.4f}',flush=True)
    table(output/'comparisons.tsv',comparisons)
    oracle={}
    for name,prob in [('image',arrays['p_clip_six']),('legacy_mm',arrays['p_legacy_mm'])]:
        pred=hierarchy(binary_truth.astype(str),prob,list('012345')).astype(int)
        oracle[name]=metric(truth,pred,list(range(6)))
    write_json(output/'oracle_diagnostic.json',{'scope':'Uses true binary labels; not a deployable method',**oracle})
    write_json(output/'routing.json',{key:{'binary_accuracy':float((np.asarray(gate)==binary_truth).mean()),
                    'fake_recall':float((np.asarray(gate)[binary_truth==0]==0).mean()) if (binary_truth==0).any() else None,
                    'real_recall':float((np.asarray(gate)[binary_truth==1]==1).mean()) if (binary_truth==1).any() else None}
                    for key,gate in gates.items()})
    np.savez_compressed(output/'replay_scores.npz',ids=np.asarray([r['id'] for r in rows]),labels=truth,
        binary_labels=binary_truth,image_features=features,title_features=titles,
        binary_raw=binary,six_raw=six,text_six_raw=text_six,**arrays)
    plan['elapsed_seconds']=time.perf_counter()-start
    return publish(output,records,predictions,details,plan)
