"""Isolated, sequential DTU pilot; each mode has its own extension and outputs."""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--dtu',required=True)
p.add_argument('--official',required=True)
p.add_argument('--output',required=True)
p.add_argument('--scans',nargs='+',default=['24','37'])
p.add_argument('--modes',nargs='+',type=int,default=list(range(6)))
p.add_argument('--iterations',type=int,default=30000)
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
names=['linear','beer','heightfield','generalized','sggx','gaussian_cdf']
scales=[1.,1.053605156578263,1.053605156578263,1.053605156578263,1.053605156578263,13.79824838]
state={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'config':vars(a),'results':[]}
def save():
 tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'status.json')
def run(cmd,cwd,env,log):
 print('RUN',*map(str,cmd),flush=True)
 with open(log,'w') as f:
  subprocess.run(list(map(str,cmd)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
for mode in a.modes:
 if mode not in range(6):raise ValueError(mode)
 root=out/f'{mode}_{names[mode]}'
 root.mkdir(exist_ok=False)
 ext=root/'extension'
 shutil.copytree(repo/'submodules/diff-surfel-rasterization',ext,ignore=shutil.ignore_patterns('build','*.so','*.egg-info','__pycache__','.git'))
 env=dict(os.environ,FOOTPRINT_MODE=str(mode),FOOTPRINT_SCALE=str(scales[mode]),MAX_JOBS='4',PYTHONPATH=str(ext)+os.pathsep+str(repo))
 env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
 record={'mode':mode,'name':names[mode],'scale':scales[mode],'state':'building'};state['results'].append(record);save()
 try:
  run([sys.executable,'setup.py','build_ext','--inplace'],ext,env,root/'build.log')
  run([sys.executable,ext/'tests/gpu_smoke_test.py'],repo,env,root/'smoke.log')
  record['state']='running';record['scans']=[];save()
  for scan in a.scans:
   target=root/f'scan{scan}';evaluation=target/'evaluation'
   item={'scan':scan,'state':'training'};record['scans'].append(item);save();start=time.time()
   run([sys.executable,'train.py','-s',Path(a.dtu)/f'scan{scan}','-m',target,'--quiet','--test_iterations','-1','--iterations',a.iterations,'--save_iterations',a.iterations,'--depth_ratio','1.0','-r','2','--lambda_dist','1000'],repo,env,root/f'scan{scan}_train.log')
   item['training_seconds']=time.time()-start;item['state']='rendering';save()
   run([sys.executable,'render.py','--iteration',a.iterations,'-s',Path(a.dtu)/f'scan{scan}','-m',target,'--quiet','--skip_train','--depth_ratio','1.0','--num_cluster','1','--voxel_size','0.004','--sdf_trunc','0.016','--depth_trunc','3.0'],repo,env,root/f'scan{scan}_render.log')
   item['state']='evaluating';save()
   run([sys.executable,'scripts/eval_dtu/evaluate_single_scene.py','--input_mesh',target/f'train/ours_{a.iterations}/fuse_post.ply','--scan_id',scan,'--output_dir',evaluation,'--mask_dir',a.dtu,'--DTU',a.official],repo,env,root/f'scan{scan}_evaluate.log')
   # The legacy evaluation wrapper uses os.system; require actual metrics.
   item['metrics']=json.loads((evaluation/'results.json').read_text());item['state']='complete';save()
  record['state']='complete'
 except Exception as e:
  record['state']='failed';record['error']=str(e)
 save()
state['finished']=True;save()
