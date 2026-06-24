"""
Scenario handling
------------------------------------
  Class-IL  → CLOPSTaskStream           (label offset, growing head)
              5 tasks × 2 cardiac classes each (10 classes total)

  Time-IL   → CLOPSSharedLabelTaskStream (shared labels, fixed 4-class head)
              3 tasks × Term 1 / Term 2 / Term 3 temporal splits

Only base CL agents are run (no sleep-CL); see run_all_chapman_sleep_v3.py for
the sleep-CL companion runner.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
_RUN_SCRIPT         = os.path.join(_SCRIPT_DIR, 'run_physionet2020_v2.py')
_DEFAULT_BASEPATH   = '/mnt/CONIRepo/ykkim/SLCL/datasets/Chapman/patient_data/'
_DEFAULT_RESULT_DIR = os.path.join(_SCRIPT_DIR, 'tscil_results', 'chapman_v3')

# Chapman: Class-IL and Time-IL only (no Domain-IL pickle available)
# GR is included for base runners (sleep runner excludes GR per the dataset note)
ALL_AGENTS    = ['LwF', 'MAS', 'SI', 'ER', 'EWC', 'DER', 'CLOPS', 'ASER', 'GR']
ALL_SCENARIOS = ['Class-IL', 'Time-IL']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _exp_name(agent: str, scenario: str, dataset: str = 'chapman') -> str:
    """Experiment folder name — must match run_physionet2020_v2.py's naming."""
    tag = scenario.replace('-', '')
    return 'v2_{}_{}_{}'.format(agent, dataset, tag)


def _result_exists(result_dir: str, agent: str, scenario: str) -> bool:
    name = _exp_name(agent, scenario)
    pkl  = os.path.join(result_dir, 'exp', name, 'result.pkl')
    return os.path.isfile(pkl)


def _build_cmd(agent: str, scenario: str, cfg) -> list:
    return [
        sys.executable, _RUN_SCRIPT,
        '--agent',          agent,
        '--cl_scenario',    scenario,
        '--dataset',        'chapman',
        '--basepath',       cfg.basepath,
        '--runs',           str(cfg.runs),
        '--epochs',         str(cfg.epochs),
        '--batch_size',     str(cfg.batch_size),
        '--lr',             str(cfg.lr),
        '--mem_budget',     str(cfg.mem_budget),
        '--max_mem_size',   str(cfg.max_mem_size),   # [C6] 1000
        '--early_stop',     str(cfg.early_stop),
        '--patience',       str(cfg.patience),
        '--lambda_impt',    str(cfg.lambda_impt),
        '--seed',           str(cfg.seed),
        '--result_dir',     cfg.result_dir,
        '--verbose',        'False',
    ]


def _run_one(cmd: list, tag: str, log_dir: str, dry_run: bool) -> dict:
    log_path = os.path.join(log_dir, '{}.log'.format(tag))

    if dry_run:
        print('[dry-run] {}'.format(' '.join(cmd)))
        return dict(tag=tag, status='skipped', elapsed_sec=0.0,
                    return_code=None, log_path=log_path)

    t0 = time.time()
    try:
        with open(log_path, 'w') as flog:
            flog.write('CMD: {}\n'.format(' '.join(cmd)))
            flog.write('START: {}\n\n'.format(datetime.now().isoformat()))
            flog.flush()
            proc = subprocess.run(cmd, stdout=flog, stderr=subprocess.STDOUT,
                                  cwd=_SCRIPT_DIR)
        elapsed = time.time() - t0
        rc      = proc.returncode
        status  = 'ok' if rc == 0 else 'error'
    except Exception as exc:
        elapsed = time.time() - t0
        rc, status = -1, 'exception'
        with open(log_path, 'a') as flog:
            flog.write('\nEXCEPTION: {}\n'.format(exc))

    return dict(tag=tag, status=status, elapsed_sec=round(elapsed, 1),
                return_code=rc, log_path=log_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Batch runner: Chapman v3 (base CL agents, C6 buffer fix)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--scenarios', nargs='+', default=ALL_SCENARIOS,
                        choices=ALL_SCENARIOS)
    parser.add_argument('--agents',    nargs='+', default=ALL_AGENTS)

    # Dataset
    parser.add_argument('--basepath',     type=str, default=_DEFAULT_BASEPATH)
    # Training
    parser.add_argument('--runs',         type=int,   default=3)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--mem_budget',   type=float, default=0.05)
    parser.add_argument('--max_mem_size', type=int,   default=1000,   # [C6]
                        help='[C6] Increased from 500; critical for Chapman pair 2-3 (22 samples)')
    parser.add_argument('--early_stop',   type=str,   default='True')
    parser.add_argument('--patience',     type=int,   default=20)
    parser.add_argument('--lambda_impt',  type=float, default=10000.0)
    parser.add_argument('--seed',         type=int,   default=1234)
    # Output
    parser.add_argument('--result_dir',  type=str, default=_DEFAULT_RESULT_DIR)
    # Behaviour
    parser.add_argument('--resume',  type=str, default='True')
    parser.add_argument('--dry_run', type=str, default='False')

    cfg = parser.parse_args()
    cfg.resume  = cfg.resume.lower()  in ('true', '1', 'yes')
    cfg.dry_run = cfg.dry_run.lower() in ('true', '1', 'yes')

    log_dir      = os.path.join(cfg.result_dir, 'logs')
    summary_path = os.path.join(cfg.result_dir, 'run_all_chapman_v3_summary.csv')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ── Build experiment queue ────────────────────────────────────────────────
    queue = []
    for scenario in cfg.scenarios:
        for agent in cfg.agents:
            tag = _exp_name(agent, scenario)
            if cfg.resume and _result_exists(cfg.result_dir, agent, scenario):
                print('[skip] {} — result exists'.format(tag))
                continue
            queue.append((tag, _build_cmd(agent, scenario, cfg)))

    total = len(queue)
    print('\n========================================')
    print(' Chapman v3 Batch Experiments (base CL)')
    print(' Scenarios : {}'.format(cfg.scenarios))
    print(' Agents    : {}'.format(cfg.agents))
    print(' Queue     : {} experiments'.format(total))
    print(' Result dir: {}'.format(cfg.result_dir))
    print('----------------------------------------')
    print(' Scenario handling:')
    print('  Class-IL  → CLOPSTaskStream (label offset, growing head)')
    print('  Time-IL   → CLOPSSharedLabelTaskStream (shared 4-class head)')
    print(' Fixes vs v2:')
    print('  [C6] max_mem_size={} (was 500)'.format(cfg.max_mem_size))
    print('========================================\n')

    if total == 0:
        print('Nothing to run. Use --resume False to re-run.')
        return

    results     = []
    batch_start = time.time()

    for done, (tag, cmd) in enumerate(queue, 1):
        print('[{:>3}/{:>3}]  {}  ...'.format(done, total, tag), flush=True)

        res = _run_one(cmd, tag, log_dir, cfg.dry_run)
        results.append(res)

        elapsed_batch = time.time() - batch_start
        remaining     = (elapsed_batch / done) * (total - done) if done else 0
        sym = {'ok': '✓', 'error': '✗', 'exception': '!',
               'skipped': '-'}.get(res['status'], '?')
        print('  {} {:>8.1f}s  (batch: {:.0f}s, ~{:.0f}s left)'.format(
            sym, res['elapsed_sec'], elapsed_batch, remaining))

        write_header = not os.path.isfile(summary_path)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(res.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(res)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - batch_start
    ok_count   = sum(1 for r in results if r['status'] == 'ok')
    err_count  = sum(1 for r in results if r['status'] not in ('ok', 'skipped'))

    print('\n========================================')
    print(' Batch done in {:.0f}s'.format(total_time))
    print(' OK      : {}'.format(ok_count))
    print(' Failed  : {}'.format(err_count))
    print(' Summary : {}'.format(summary_path))
    print('========================================\n')

    if err_count:
        print('Failed experiments:')
        for r in results:
            if r['status'] not in ('ok', 'skipped'):
                print('  {}  →  rc={}  log={}'.format(
                    r['tag'], r['return_code'], r['log_path']))

    json_path = summary_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump({'total_time_sec': total_time,
                   'ok': ok_count,
                   'failed': err_count,
                   'experiments': results}, f, indent=2)
    print('JSON summary saved to:', json_path)


if __name__ == '__main__':
    main()
