"""Run CSV and checkpoint analyses together, with separate logs and outputs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import class_analysis_common as common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-root',type=Path,required=True)
    p.add_argument('--meta-root',type=Path,default=common.HERE.parent/'data2/balance')
    p.add_argument('--archive-root',type=Path,default=common.HERE)
    p.add_argument('--output',type=Path)
    p.add_argument('--metrics-seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--geometry-seeds',type=int,nargs='+',default=[42])
    p.add_argument('--steps',type=int,nargs='+',default=[1,5,9])
    p.add_argument('--reference-per-class',type=int,default=20)
    p.add_argument('--query-per-class',type=int,default=0)
    p.add_argument('--sample-seed',type=int,default=20261001)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if args.resume and args.output is None:
        p.error('--resume requires the previous --output directory')
    args.output = args.output or common.HERE/'results'/('class_benefit_geometry_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    # Cheap common preflight; no training imports or feature mutations.
    common.discover_runs(args.archive_root,sorted(set(args.metrics_seeds+args.geometry_seeds)))
    for file in [args.feature_root/'visual_features.h5',args.feature_root/'audio_pretrained_feature/audio_pretrained_feature_dict.npy']:
        if not common.native(file).is_file():
            raise FileNotFoundError(str(file))
    options = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['resume','output']}
    out = common.prepare_output(args.output,options,args.resume)
    commands = {
        'metrics':[sys.executable,'-u',str(common.HERE/'analyze_class_benefits.py'),
            '--archive-root',str(args.archive_root),'--seeds',*map(str,args.metrics_seeds),'--output',str(out/'metrics')],
        'geometry':[sys.executable,'-u',str(common.HERE/'probe_class_geometry.py'),
            '--archive-root',str(args.archive_root),'--feature-root',str(args.feature_root),'--meta-root',str(args.meta_root),
            '--seeds',*map(str,args.geometry_seeds),'--steps',*map(str,args.steps),
            '--reference-per-class',str(args.reference_per_class),'--query-per-class',str(args.query_per_class),
            '--sample-seed',str(args.sample_seed),'--threads',str(args.threads),'--batch-size',str(args.batch_size),
            '--device',args.device,'--output',str(out/'geometry')]}
    if args.resume:
        for command in commands.values():
            command.append('--resume')
    common.write_json(out/'commands.json',commands)
    print('Output:',out,flush=True)
    processes,logs,exit_codes = {},{},{}
    try:
        for name,command in commands.items():
            logs[name] = (out/(name+'.log')).open('a' if args.resume else 'w',encoding='utf-8')
            processes[name] = subprocess.Popen(command,stdout=logs[name],stderr=subprocess.STDOUT)
            print(f'{name}: PID {processes[name].pid}, log {out/(name+".log")}',flush=True)
        last = time.monotonic()
        while len(exit_codes)<len(processes):
            for name,process in processes.items():
                if name not in exit_codes and process.poll() is not None:
                    exit_codes[name] = process.returncode
                    print(f'{name} exited with code {process.returncode}',flush=True)
            if time.monotonic()-last>30:
                print('Running:',', '.join(k for k in processes if k not in exit_codes),flush=True)
                last = time.monotonic()
            if len(exit_codes)<len(processes):
                time.sleep(1)
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        raise
    finally:
        for log in logs.values():
            log.close()
    common.write_json(out/'status.json',exit_codes)
    if any(exit_codes.values()):
        raise SystemExit('One or more analyses failed; inspect logs. Successful outputs are preserved.')
    print('Both analyses complete:',out,flush=True)


if __name__ == '__main__':
    main()
