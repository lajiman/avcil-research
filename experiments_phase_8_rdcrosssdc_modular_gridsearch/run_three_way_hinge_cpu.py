"""Run the two previously measured hinge settings sequentially on a CPU.

No Slurm, training, checkpoint writes, or manual lambda overrides required.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from probe_checkpoint_losses import native
from probe_checkpoint_three_way import find_metrics_dir

HERE = Path(__file__).resolve().parent
RUNS = [
    ('lambda_0p03_alpha_0p5', 'VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_tol_focus_v1_hinge_A_lc0p03_a0p5_tol0p01_s1p0_h200_seed42'),
    ('lambda_0p1_alpha_1p0', 'VGGSound_random_balance_rd_crosssdc_replace_c_trust_only_tol_focus_v1_hinge_B_lc0p1_a1p0_tol0p01_s1p0_h200_seed42'),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', type=Path, required=True)
    parser.add_argument('--meta-root', type=Path, default=HERE.parent/'data2'/'balance')
    parser.add_argument('--archive-root', type=Path, default=HERE/'save_commands_cmr_hinge_tolerance_focus_3seeds')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--steps', type=int, nargs='+', default=[1,5,9])
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--probe-seed', type=int, default=42)
    parser.add_argument('--loss-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Check checkpoint paths and print commands without computing')
    cli = parser.parse_args(argv)
    out = (cli.output or HERE/'results'/f'three_way_hinge_cpu_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    if native(out).exists() and (not native(out).is_dir() or any(native(out).iterdir())):
        parser.error('Output directory must be new or empty')
    if cli.batches < 1 or cli.threads < 1 or any(s < 1 for s in cli.steps) or len(set(cli.steps)) != len(cli.steps):
        parser.error('Positive batches/threads and unique steps >= 1 required')
    for _, run in RUNS:
        if out.is_relative_to((cli.archive_root/run).resolve()):
            parser.error('Analysis output must be outside the checkpoint directories')
        find_metrics_dir(cli.archive_root/run)
        for step in {s for t in cli.steps for s in [t,t-1]}:
            path = cli.archive_root/run/f'step_{step}_best_model.pkl'
            if not native(path).is_file():
                parser.error(f'Missing checkpoint: {path}; use --archive-root if necessary')
    commands = []
    for index, (label, run) in enumerate(RUNS):
        command = [sys.executable, '-u', str(HERE/'probe_checkpoint_three_way.py'),
            '--run-dir', str(cli.archive_root/run), '--feature-root', str(cli.feature_root),
            '--meta-root', str(cli.meta_root), '--output', str(out/label), '--device', 'cpu',
            '--threads', str(cli.threads), '--probe-seed', str(cli.probe_seed),
            '--steps', *map(str, cli.steps), '--batches', str(cli.batches)]
        if index:
            command += ['--probe-manifest', str(out/RUNS[0][0]/'probe_manifest.json')]
        if cli.loss_only:
            command.append('--loss-only')
        commands.append((label, command))
    print('Results:', out, flush=True)
    if cli.dry_run:
        for _, command in commands:
            print(shlex.join(command))
        return
    native(out).mkdir(parents=True, exist_ok=True)
    batch = dict(status='running', runs=[], cli=vars(cli))
    def save_status():
        native(out/'batch_metadata.json').write_text(json.dumps(batch, indent=2, default=str), encoding='utf-8')
    save_status()
    try:
        for label, command in commands:
            print(f'Running {label}; lambda is read from its checkpoint', flush=True)
            with native(out/(label+'.log')).open('w', encoding='utf-8') as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding='utf-8', errors='replace',
                        env=dict(os.environ, PYTHONIOENCODING='utf-8')) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
            if code:
                raise RuntimeError(f'{label} failed (exit {code}); see {out/(label+".log")}')
            metadata = json.loads(native(out/label/'metadata.json').read_text(encoding='utf-8'))
            if metadata['status'] != 'complete':
                raise RuntimeError(f'{label} did not finish all measurements')
            batch['runs'].append(dict(run=label, status='complete', output=str(out/label)))
            save_status()
        batch['status'] = 'complete'
    except Exception as exc:
        batch['status'], batch['error'] = 'failed', str(exc)
        raise
    finally:
        save_status()
    print('Both settings completed:', out, flush=True)


if __name__ == '__main__':
    main()
