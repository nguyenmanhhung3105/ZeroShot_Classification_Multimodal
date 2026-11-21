"""Standalone dev-only notebook for controlled Fakeddit hierarchy evaluation."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def prepare(mode='smoke',output=None,variant='hierarchy'):
    if variant not in ['hierarchy','stata_calip']:
        raise ValueError('Unsupported Fakeddit variant')
    calip=variant=='stata_calip'
    if mode not in ['smoke','dev']:
        raise ValueError('No test job in this development experiment')
    names=['run.py','requirements.txt','configs/project.yaml','configs/fakeddit_prompts.json',
           'configs/fakeddit_hierarchy_dev.json','src/vlm_project/__init__.py',
           *[f'src/vlm_project/{name}.py' for name in ['fakeddit_hierarchy','adaptation','methods','data','backbone','io','evaluate']]]
    if calip:
        names.extend(['src/vlm_project/calip.py','src/vlm_project/fakeddit_calip_ablation.py','configs/fakeddit_calip_ablation.json'])
    files={name:(ROOT/name).read_text() for name in names}
    hashes={name:hashlib.sha256(content.encode()).hexdigest() for name,content in files.items()}
    destination=Path(output or ROOT/'results/_system/fakeddit_hierarchy'/mode)
    destination.mkdir(parents=True,exist_ok=True)
    bootstrap=f'''import json, subprocess, sys, zipfile
from pathlib import Path
files=json.loads({json.dumps(files)!r})
PROJECT=Path('/kaggle/working/fakeddit_hierarchy_source')
for name,content in files.items():
    file=PROJECT/name
    if file.exists() and file.read_text()!=content:
        raise ValueError('Existing source differs; start a clean session')
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(content)
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(PROJECT/'requirements.txt')],check=True)
sys.path.insert(0,str(PROJECT/'src'))
from vlm_project.io import config, locate_data
from vlm_project.fakeddit_hierarchy import run
settings=config(PROJECT/'configs/project.yaml')
settings['device']='cuda'
settings['output_root']='/kaggle/working/fakeddit_hierarchy_results'
'''
    execute=f'''root=locate_data('fakeddit','dev',settings)
output=run(settings,'dev',root,smoke={mode=='smoke'!r},calip={calip!r})
result_zip=Path('/kaggle/working')/('metrics_'+output.name+'.zip')
with zipfile.ZipFile(result_zip,'w',zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(output.iterdir()):
        if file.is_file():
            archive.write(file,'run/'+file.name)
print('Metrics ZIP:',result_zip)
'''
    cells=[]
    for kind,text in [('markdown','# Fakeddit dev 6way: matched hierarchy controls and original multimodal reference'),('code',bootstrap),('code',execute)]:
        cell={'cell_type':kind,'id':f'fakeddit-hierarchy-{len(cells)}','metadata':{},'source':text.splitlines(True)}
        if kind=='code':
            compile(text,'fakeddit-generated','exec')
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}},'cells':cells}
    slug='stata-calip' if calip else 'hierarchy'
    metadata={'id':f'nguyenhung3105/vlm-fakeddit-{slug}-{mode}-v1','title':f'VLM Fakeddit {slug.title()} {mode.title()} V1',
              'code_file':'run.ipynb','language':'python','kernel_type':'notebook','is_private':'true',
              'enable_gpu':'true','enable_internet':'true','machine_shape':'NvidiaTeslaT4',
              'dataset_sources':['nguyenhung3105/vlm-fakeddit-dev-v2'],'competition_sources':[],'kernel_sources':[]}
    for name,value in [('run.ipynb',notebook),('kernel-metadata.json',metadata),('source_manifest.json',{'files':hashes,'mode':mode})]:
        (destination/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return destination
