"""
Batch runner: Chapman CL + sleep-CL.

Scenario handling
-------------------------------------
  Class-IL  → CLOPSTaskStream           (label offset, growing head)
              5 tasks × 2 cardiac classes each (10 classes total)

  Time-IL   → CLOPSSharedLabelTaskStream (shared labels, fixed 4-class head)
              3 tasks × Term 1 / Term 2 / Term 3 temporal splits

Results saved to tscil_results/chapman_v4/ (separate from v3).
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
_RUN_SCRIPT         = os.path.join(_SCRIPT_DIR, 'run_physionet2020_chapman_v4.py')
_DEFAULT_BASEPATH   = '/mnt/CONIRepo/ykkim/SLCL/datasets/Chapman/patient_data/'
_DEFAULT_RESULT_DIR = os.path.join(_SCRIPT_DIR, 'tscil_results', 'chapman_v4')

ALL_AGENTS    = ['LwF', 'MAS', 'SI', 'ER', 'EWC', 'DER', 'CLOPS', 'ASER', 'GR']
ALL_SCENARIOS = ['Class-IL', 'Time-IL']

SCENARIO_HPARAMS = {
    'Class-IL': {
        'replace_bn_with_gn':         False,
        'down_alpha':                  0.92,
        'sleep_steps':                  150,
        'sleep_steps_min':              100,
        'sleep_steps_max':              200,
        'spindle_base_steps_min':         5,
        'spindle_base_steps_max':        15,
        'lambda_distill':               0.0,
        'lambda_distill_min':          -1.0,
        'lambda_distill_max':          -1.0,
        'merge_lam':                   0.40,
        'merge_decay':                 0.00,
        'merge_lam_floor':             0.30,
        'lambda_div':                  0.10,
        'lambda_old':                  1.00,
        'fisher_protect_quantile':     0.80,
        'alpha_protect':               0.97,
        'focused_steps_per_task':        10,
        'skip_phase1_threshold':       0.995,
        'use_rollback':                True,
        'rollback_threshold':          0.70,  
        'rollback_n_batches':            20,
        'downreg_stabilize_steps':       10,
        'use_post_downreg_teacher':    True,
    },
    'Time-IL': {
        'replace_bn_with_gn':         False,
        'down_alpha':                  0.95,
        'sleep_steps':                  100,
        'sleep_steps_min':               70,
        'sleep_steps_max':              150,
        'spindle_base_steps_min':         5,
        'spindle_base_steps_max':        15,
        'lambda_distill':               0.0,
        'lambda_distill_min':          -1.0,
        'lambda_distill_max':          -1.0,
        'merge_lam':                   0.30,
        'merge_decay':                 0.15,
        'merge_lam_floor':             0.08,
        'lambda_div':                  0.10,
        'lambda_old':                  0.20,
        'fisher_protect_quantile':     0.80,
        'alpha_protect':               0.97,
        'focused_steps_per_task':         8,
        'skip_phase1_threshold':       0.995,
        'use_rollback':                True,
        'rollback_threshold':          0.80,  
        'rollback_n_batches':            20,
        'downreg_stabilize_steps':       10,
        'use_post_downreg_teacher':    True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _exp_name(agent: str, scenario: str, sleep_base: str = None,
              dataset: str = 'chapman') -> str:
    """Experiment folder name — must match run_physionet2020_chapman_v4.py naming."""
    tag = scenario.replace('-', '')
    if agent == 'sleep-CL':
        return 'v4c_sleepCL_{}_{}_{}'.format(sleep_base, dataset, tag)
    return 'v4c_{}_{}_{}'.format(agent, dataset, tag)


def _result_exists(result_dir: str, agent: str, scenario: str,
                   sleep_base: str = None) -> bool:
    name = _exp_name(agent, scenario, sleep_base)
    pkl  = os.path.join(result_dir, 'exp', name, 'result.pkl')
    return os.path.isfile(pkl)


def _build_base_cmd(agent: str, scenario: str, cfg) -> list:
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
        '--max_mem_size',   str(cfg.max_mem_size),
        '--early_stop',     str(cfg.early_stop),
        '--patience',       str(cfg.patience),
        '--lambda_impt',    str(cfg.lambda_impt),
        '--seed',           str(cfg.seed),
        '--result_dir',     cfg.result_dir,
        '--verbose',        'False',
    ]


def _build_sleep_cmd(sleep_base: str, scenario: str, cfg) -> list:
    sc = SCENARIO_HPARAMS.get(scenario, {})

    def _hp(key, fallback=None):
        return sc[key] if key in sc else getattr(cfg, key, fallback)

    return [
        sys.executable, _RUN_SCRIPT,
        '--agent',                    'sleep-CL',
        '--sleep_base_agent',         sleep_base,
        '--cl_scenario',              scenario,
        '--dataset',                  'chapman',
        '--basepath',                 cfg.basepath,
        '--runs',                     str(cfg.runs),
        '--epochs',                   str(cfg.epochs),
        '--batch_size',               str(cfg.batch_size),
        '--lr',                       str(cfg.lr),
        '--mem_budget',               str(cfg.mem_budget),
        '--max_mem_size',             str(cfg.max_mem_size),
        '--early_stop',               str(cfg.early_stop),
        '--patience',                 str(cfg.patience),
        '--lambda_impt',              str(cfg.lambda_impt),
        '--seed',                     str(cfg.seed),
        '--result_dir',               cfg.result_dir,
        '--verbose',                  'False',
        # ── sleep-CL hyperparameters (scenario-aware) ─────────────────────
        '--sleep_steps',              str(_hp('sleep_steps',                150)),
        '--sleep_lr',                 str(cfg.sleep_lr),
        '--down_alpha',               str(_hp('down_alpha',               0.92)),
        '--merge_lam',                str(_hp('merge_lam',                0.40)),
        '--lambda_distill',           str(_hp('lambda_distill',           0.0)),
        '--lambda_div',               str(_hp('lambda_div',               0.10)),
        '--lambda_old',               str(_hp('lambda_old',               1.00)),
        '--fisher_protect_quantile',  str(_hp('fisher_protect_quantile',  0.80)),
        '--alpha_protect',            str(_hp('alpha_protect',            0.97)),
        '--focused_steps_per_task',   str(_hp('focused_steps_per_task',     10)),
        '--sleep_samples_per_class',  str(cfg.sleep_samples_per_class),
        # [C2] GroupNorm disabled for ECG
        '--replace_bn_with_gn',       str(_hp('replace_bn_with_gn',       False)),
        '--sleep_steps_min',          str(_hp('sleep_steps_min',             100)),
        '--sleep_steps_max',          str(_hp('sleep_steps_max',             200)),
        '--lambda_distill_min',       str(_hp('lambda_distill_min',         -1.0)),
        '--lambda_distill_max',       str(_hp('lambda_distill_max',         -1.0)),
        '--spindle_base_steps_min',   str(_hp('spindle_base_steps_min',        5)),
        '--spindle_base_steps_max',   str(_hp('spindle_base_steps_max',       15)),
        # ── Alt 2 / Alt 4 tuning ──────────────────────────────────────────
        '--skip_phase1_threshold',    str(_hp('skip_phase1_threshold',    0.995)),
        '--merge_decay',              str(_hp('merge_decay',               0.00)),
        '--merge_lam_floor',          str(_hp('merge_lam_floor',           0.30)),
        # ── [V4-3/4/5] BN-safe rollback ───────────────────────────────────
        '--use_rollback',             str(_hp('use_rollback',              True)),
        '--rollback_threshold',       str(_hp('rollback_threshold',        0.80)),
        '--rollback_n_batches',       str(_hp('rollback_n_batches',          20)),
        '--aser_rollback_factor',     str(cfg.aser_rollback_factor),
        # ── [V4-1/2] Phase 1.5 + Method 2 ────────────────────────────────
        '--downreg_stabilize_steps',  str(_hp('downreg_stabilize_steps',    10)),
        '--use_post_downreg_teacher', str(_hp('use_post_downreg_teacher', True)),
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
        description='Batch runner: Chapman V4 (Phase 1.5 + Method 2 + BN rollback)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--groups',    nargs='+', default=['base', 'sleep'],
                        choices=['base', 'sleep'])
    parser.add_argument('--scenarios', nargs='+', default=ALL_SCENARIOS,
                        choices=ALL_SCENARIOS)
    parser.add_argument('--agents',    nargs='+', default=ALL_AGENTS,
                        help='Unified list — base agents AND sleep-CL wake-phase bases')

    # Dataset
    parser.add_argument('--basepath',     type=str, default=_DEFAULT_BASEPATH)
    # Training
    parser.add_argument('--runs',         type=int,   default=5)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--mem_budget',   type=float, default=0.05)
    parser.add_argument('--max_mem_size', type=int,   default=1000,
                        help='[C6] 1000 — critical for pair 2-3 (22 samples)')
    parser.add_argument('--early_stop',   type=str,   default='True')
    parser.add_argument('--patience',     type=int,   default=20)
    parser.add_argument('--lambda_impt',  type=float, default=10000.0)
    parser.add_argument('--seed',         type=int,   default=1234)
    # sleep-CL defaults (overridden per-scenario by SCENARIO_HPARAMS)
    parser.add_argument('--sleep_lr',                type=float, default=1e-3)
    parser.add_argument('--lambda_distill',          type=float, default=0.0,
                        help='[C5] Default 0.0 — KL distill harmful for ECG')
    parser.add_argument('--fisher_protect_quantile', type=float, default=0.80)
    parser.add_argument('--alpha_protect',           type=float, default=0.97)
    parser.add_argument('--sleep_samples_per_class', type=int,   default=50,
                        help='[C6] 50 — critical for pair 2-3')
    parser.add_argument('--replace_bn_with_gn',      type=str,   default='False',
                        help='[C2] Disabled — GN harmful for ECG')
    # [V4-5] ASER rollback factor (global — same for all scenarios)
    parser.add_argument('--aser_rollback_factor',    type=float, default=0.75,
                        help='[V4-5] Scales rollback threshold for ASER hard-example buffer')
    # Output
    parser.add_argument('--result_dir',  type=str, default=_DEFAULT_RESULT_DIR)
    # Behaviour
    parser.add_argument('--resume',  type=str, default='True')
    parser.add_argument('--dry_run', type=str, default='False')

    cfg = parser.parse_args()
    cfg.resume  = cfg.resume.lower()  in ('true', '1', 'yes')
    cfg.dry_run = cfg.dry_run.lower() in ('true', '1', 'yes')

    log_dir      = os.path.join(cfg.result_dir, 'logs')
    summary_path = os.path.join(cfg.result_dir, 'run_all_chapman_sleep_v4_summary.csv')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ── Build experiment queue ────────────────────────────────────────────────
    queue = []

    for scenario in cfg.scenarios:
        if 'base' in cfg.groups:
            for agent in cfg.agents:
                tag = _exp_name(agent, scenario)
                if cfg.resume and _result_exists(cfg.result_dir, agent, scenario):
                    print('[skip] {} — result exists'.format(tag))
                    continue
                queue.append((tag, _build_base_cmd(agent, scenario, cfg)))

        if 'sleep' in cfg.groups:
            for agent in cfg.agents:
                tag = _exp_name('sleep-CL', scenario, sleep_base=agent)
                if cfg.resume and _result_exists(
                        cfg.result_dir, 'sleep-CL', scenario, agent):
                    print('[skip] {} — result exists'.format(tag))
                    continue
                queue.append((tag, _build_sleep_cmd(agent, scenario, cfg)))

    total = len(queue)
    print('\n========================================')
    print(' Chapman Sleep-CL V4 Batch Experiments')
    print(' Scenarios : {}'.format(cfg.scenarios))
    print(' Groups    : {}'.format(cfg.groups))
    print(' Agents    : {}'.format(cfg.agents))
    print(' Queue     : {} experiments'.format(total))
    print(' Result dir: {}'.format(cfg.result_dir))
    print('----------------------------------------')
    print(' Scenario handling:')
    print('  Class-IL  → CLOPSTaskStream (label offset, growing head)')
    print('  Time-IL   → CLOPSSharedLabelTaskStream (shared 4-class head)')
    print(' Changes vs v3:')
    print('  [V4-1] Phase 1.5: downreg_stabilize_steps=10')
    print('  [V4-2] Method 2:  use_post_downreg_teacher=True')
    print('  [V4-3] BN-safe rollback (state_dict — params + BN buffers)')
    print('  [V4-4] Rollback threshold: Class-IL=0.70, Time-IL=0.80')
    print('  [V4-5] ASER rollback factor={:.2f}'.format(cfg.aser_rollback_factor))
    print(' Fixes from v3:')
    print('  [C1] Time-IL: lambda_old=0.20, merge_lam=0.30, merge_decay=0.15')
    print('  [C2] replace_bn_with_gn=False (GN harmful for ECG)')
    print('  [C5] ECG hparams (down_alpha, sleep_steps, lambda_distill=0, etc.)')
    print('  [C6] max_mem_size={}, sleep_samples_per_class={}'.format(
        cfg.max_mem_size, cfg.sleep_samples_per_class))
    if 'sleep' in cfg.groups:
        print(' Sleep-CL V4 hyperparameters per scenario:')
        for sc in cfg.scenarios:
            hp = SCENARIO_HPARAMS.get(sc, {})
            print('  [{:<9}]  down_alpha={:.3f}  lambda_old={:.2f}  '
                  'merge_lam={:.2f} (decay={:.2f}, floor={:.2f})  '
                  'sleep_steps={:d}  rollback_thr={:.2f}  '
                  'stab_steps={:d}  post_teacher={}'.format(
                      sc,
                      hp.get('down_alpha',              0.92),
                      hp.get('lambda_old',              1.00),
                      hp.get('merge_lam',               0.40),
                      hp.get('merge_decay',             0.00),
                      hp.get('merge_lam_floor',         0.30),
                      hp.get('sleep_steps',              150),
                      hp.get('rollback_threshold',       0.80),
                      hp.get('downreg_stabilize_steps',   10),
                      hp.get('use_post_downreg_teacher', True)))
    print('========================================\n')

    if total == 0:
        print('Nothing to run. Use --resume False to re-run.')
        return

    results     = []
    batch_start = time.time()

    for done, (tag, cmd) in enumerate(queue, start=1):
        print('[{:>3}/{:>3}]  {}  ...'.format(done, total, tag), flush=True)
        res = _run_one(cmd, tag, log_dir, cfg.dry_run)
        results.append(res)

        elapsed_batch = time.time() - batch_start
        remaining     = (elapsed_batch / done) * (total - done)
        sym = {'ok': '+', 'error': 'X', 'exception': '!',
               'skipped': '-'}.get(res['status'], '?')
        print('  {} {:>8.1f}s  (total: {:.0f}s, ~{:.0f}s remaining)'.format(
            sym, res['elapsed_sec'], elapsed_batch, remaining))

        write_header = not os.path.isfile(summary_path)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(res.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(res)

    total_time = time.time() - batch_start
    ok_count   = sum(1 for r in results if r['status'] == 'ok')
    err_count  = sum(1 for r in results if r['status'] not in ('ok', 'skipped'))

    print('\n========================================')
    print(' Done in {:.0f}s  |  OK: {}  |  Failed: {}'.format(
        total_time, ok_count, err_count))
    print(' CSV: {}'.format(summary_path))
    print('========================================\n')

    if err_count:
        print('Failed:')
        for r in results:
            if r['status'] not in ('ok', 'skipped'):
                print('  {}  rc={}  log={}'.format(
                    r['tag'], r['return_code'], r['log_path']))

    json_path = summary_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump({'total_time_sec': total_time, 'ok': ok_count,
                   'failed': err_count, 'experiments': results}, f, indent=2)
    print('JSON:', json_path)


if __name__ == '__main__':
    main()
