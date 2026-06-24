"""
----------------------------------
Batch runner: Physionet2020 base CL + sleep-CL.

Scenario hyperparameters
-------------------------
  Class-IL  (5 tasks, offset labels)
    - rollback_threshold : 0.70  
    - rollback_n_batches : 20
    - SWR-only           : True
    - merge_lam          : 0.15 

  Time-IL   (3 tasks, shared labels)
    - auto_swr_only      : True
    - min_fgt            : 5.0%

  Domain-IL  (12 tasks, shared labels)
    - auto_swr_only      : True
    - every_n            : 3
    - min_fgt            : 2.0%
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_SCRIPT_DIR              = os.path.dirname(os.path.abspath(__file__))
_RUN_SCRIPT              = os.path.join(_SCRIPT_DIR, 'run_physionet2020_v7.py')
_DEFAULT_BASEPATH        = '/mnt/CONIRepo/ykkim/SLCL/datasets/'
_DEFAULT_BASEPATH_PHYSIO = '/mnt/CONIRepo/ykkim/SLCL/datasets/Physionet2020/patient_data/'
_DEFAULT_RESULT_DIR      = os.path.join(_SCRIPT_DIR, 'tscil_results', 'physionet2020_v7')

ALL_AGENTS    = ['LwF', 'MAS', 'SI', 'ER', 'EWC', 'DER', 'CLOPS', 'ASER', 'GR']
ALL_SCENARIOS = ['Class-IL', 'Time-IL', 'Domain-IL']


SCENARIO_HPARAMS_V7 = {
    'Class-IL': {
        'use_rollback':                True,
        'rollback_threshold':          0.70, 
        'rollback_n_batches':          20,
        'use_fisher_merge':            False,
        'swr_only':                    True,
        'auto_swr_only':               False,
        'min_fgt_to_sleep':            0.0,
        'sleep_every_n_tasks':         1,
        'merge_lam':                   0.15,
        'merge_lam_floor':             0.10,
        'merge_decay':                 0.00,
        'lambda_old':                  1.00,
        'lambda_div':                  0.10,
        'down_alpha':                  0.90,
        'skip_phase1_threshold':       0.995,
        'sleep_steps':                 150,
        'sleep_steps_min':             100,
        'sleep_steps_max':             200,
        'spindle_base_steps_min':        5,
        'spindle_base_steps_max':       15,
        'lambda_distill':              0.0,
        'lambda_distill_min':         -1.0,
        'lambda_distill_max':         -1.0,
        'fisher_protect_quantile':     0.80,
        'alpha_protect':               0.97,
        'focused_steps_per_task':      10,
        'replace_bn_with_gn':          False,
    },
    'Time-IL': {
        'use_rollback':                True,
        'rollback_threshold':          0.80,   # 유지
        'rollback_n_batches':          20,
        'use_fisher_merge':            False,
        'swr_only':                    False,
        'auto_swr_only':               True,
        'min_fgt_to_sleep':            5.0,
        'sleep_every_n_tasks':         1,
        'merge_lam':                   0.08,
        'merge_lam_floor':             0.05,
        'merge_decay':                 0.05,
        'lambda_old':                  0.05,
        'lambda_div':                  0.08,
        'down_alpha':                  0.97,
        'skip_phase1_threshold':       0.990,
        'sleep_steps':                 100,
        'sleep_steps_min':              70,
        'sleep_steps_max':             150,
        'spindle_base_steps_min':        5,
        'spindle_base_steps_max':       15,
        'lambda_distill':              0.0,
        'lambda_distill_min':         -1.0,
        'lambda_distill_max':         -1.0,
        'fisher_protect_quantile':     0.80,
        'alpha_protect':               0.97,
        'focused_steps_per_task':        8,
        'replace_bn_with_gn':          False,
    },
    'Domain-IL': {
        'use_rollback':                True,
        'rollback_threshold':          0.90,   
        'rollback_n_batches':          20,
        'use_fisher_merge':            False,
        'swr_only':                    False,
        'auto_swr_only':               True,
        'min_fgt_to_sleep':            2.0,
        'sleep_every_n_tasks':         3,
        'merge_lam':                   0.05,
        'merge_lam_floor':             0.03,
        'merge_decay':                 0.05,
        'lambda_old':                  0.05,
        'lambda_div':                  0.12,
        'down_alpha':                  0.99,
        'skip_phase1_threshold':       0.990,
        'sleep_steps':                  80,
        'sleep_steps_min':              50,
        'sleep_steps_max':             120,
        'spindle_base_steps_min':        5,
        'spindle_base_steps_max':       15,
        'lambda_distill':              0.0,
        'lambda_distill_min':         -1.0,
        'lambda_distill_max':         -1.0,
        'fisher_protect_quantile':     0.80,
        'alpha_protect':               0.97,
        'focused_steps_per_task':        8,
        'replace_bn_with_gn':          False,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _exp_name(agent: str, scenario: str, sleep_base: str = None,
              dataset: str = 'physionet2020') -> str:
    tag = scenario.replace('-', '')
    if agent == 'sleep-CL':
        return 'v7_sleepCL_{}_{}_{}'.format(sleep_base, dataset, tag)
    return 'v7_{}_{}_{}'.format(agent, dataset, tag)


def _result_exists(result_dir: str, agent: str, scenario: str,
                   sleep_base: str = None) -> bool:
    name = _exp_name(agent, scenario, sleep_base)
    pkl  = os.path.join(result_dir, 'exp', name, 'result.pkl')
    return os.path.isfile(pkl)


def _build_base_cmd(agent: str, scenario: str, cfg) -> list:
    cmd = [
        sys.executable, _RUN_SCRIPT,
        '--agent',          agent,
        '--cl_scenario',    scenario,
        '--dataset',        'physionet2020',
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
    if cfg.basepath_physionet:
        cmd += ['--basepath_physionet', cfg.basepath_physionet]
    return cmd


def _build_sleep_cmd(sleep_base: str, scenario: str, cfg) -> list:
    sc = SCENARIO_HPARAMS_V7.get(scenario, {})

    def _hp(key, fallback=None):
        return sc[key] if key in sc else getattr(cfg, key, fallback)

    cmd = [
        sys.executable, _RUN_SCRIPT,
        '--agent',                    'sleep-CL',
        '--sleep_base_agent',         sleep_base,
        '--cl_scenario',              scenario,
        '--dataset',                  'physionet2020',
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
        # ── v2 sleep params ───────────────────────────────────────────────
        '--sleep_steps',              str(_hp('sleep_steps',               150)),
        '--sleep_lr',                 str(cfg.sleep_lr),
        '--down_alpha',               str(_hp('down_alpha',               0.90)),
        '--merge_lam',                str(_hp('merge_lam',                0.15)),
        '--lambda_distill',           str(_hp('lambda_distill',            0.0)),
        '--lambda_div',               str(_hp('lambda_div',               0.10)),
        '--lambda_old',               str(_hp('lambda_old',               1.00)),
        '--fisher_protect_quantile',  str(_hp('fisher_protect_quantile',  0.80)),
        '--alpha_protect',            str(_hp('alpha_protect',            0.97)),
        '--focused_steps_per_task',   str(_hp('focused_steps_per_task',     10)),
        '--sleep_samples_per_class',  str(cfg.sleep_samples_per_class),
        '--replace_bn_with_gn',       str(_hp('replace_bn_with_gn',      False)),
        '--sleep_steps_min',          str(_hp('sleep_steps_min',           100)),
        '--sleep_steps_max',          str(_hp('sleep_steps_max',           200)),
        '--lambda_distill_min',       str(_hp('lambda_distill_min',       -1.0)),
        '--lambda_distill_max',       str(_hp('lambda_distill_max',       -1.0)),
        '--spindle_base_steps_min',   str(_hp('spindle_base_steps_min',      5)),
        '--spindle_base_steps_max',   str(_hp('spindle_base_steps_max',     15)),
        '--skip_phase1_threshold',    str(_hp('skip_phase1_threshold',   0.995)),
        '--merge_decay',              str(_hp('merge_decay',              0.00)),
        '--merge_lam_floor',          str(_hp('merge_lam_floor',          0.10)),
        # ── v3 sleep params (Alt 5/6/7) ───────────────────────────────────
        '--swr_only',                 str(_hp('swr_only',                False)),
        '--auto_swr_only',            str(_hp('auto_swr_only',            True)),
        '--sleep_every_n_tasks',      str(_hp('sleep_every_n_tasks',         1)),
        '--min_fgt_to_sleep',         str(_hp('min_fgt_to_sleep',          0.0)),
        # ── v7 sleep params (BN-safe rollback) ────────────────────────────
        '--use_rollback',             str(_hp('use_rollback',              True)),
        '--rollback_threshold',       str(_hp('rollback_threshold',        0.85)),
        '--rollback_n_batches',       str(_hp('rollback_n_batches',          20)),
        '--aser_rollback_factor',     str(cfg.aser_rollback_factor),
        '--use_fisher_merge',         str(_hp('use_fisher_merge',         False)),
        '--fisher_beta',              str(_hp('fisher_beta',               1.0)),
        '--fisher_n_batches',         str(_hp('fisher_n_batches',            10)),
    ]
    if cfg.basepath_physionet:
        cmd += ['--basepath_physionet', cfg.basepath_physionet]
    return cmd


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
        description='Physionet2020 v7 batch: sleep-CL V7 (BN-safe rollback)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--groups',    nargs='+', default=['base', 'sleep'],
                        choices=['base', 'sleep'])
    parser.add_argument('--scenarios', nargs='+', default=ALL_SCENARIOS,
                        choices=ALL_SCENARIOS)
    parser.add_argument('--agents',    nargs='+', default=ALL_AGENTS)

    # Dataset
    parser.add_argument('--basepath',            type=str, default=_DEFAULT_BASEPATH)
    parser.add_argument('--basepath_physionet',  type=str, default=_DEFAULT_BASEPATH_PHYSIO)
    # Training
    parser.add_argument('--runs',         type=int,   default=3)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--mem_budget',   type=float, default=0.05)
    parser.add_argument('--max_mem_size', type=int,   default=1500)
    parser.add_argument('--early_stop',   type=str,   default='True')
    parser.add_argument('--patience',     type=int,   default=20)
    parser.add_argument('--lambda_impt',  type=float, default=10000.0)
    parser.add_argument('--seed',         type=int,   default=1234)
    # sleep-CL global defaults
    parser.add_argument('--sleep_lr',                type=float, default=1e-3)
    parser.add_argument('--sleep_samples_per_class', type=int,   default=50)
    # V7 global param (applied for ASER across all scenarios)
    parser.add_argument('--aser_rollback_factor', type=float, default=0.75,
                        help='Scale rollback threshold for ASER (hard-example buffer'
                             ' → lower proxy acc; factor<1 prevents over-rollback)')
    # Output
    parser.add_argument('--result_dir',  type=str, default=_DEFAULT_RESULT_DIR)
    # Behaviour
    parser.add_argument('--resume',  type=str, default='True')
    parser.add_argument('--dry_run', type=str, default='False')

    cfg = parser.parse_args()
    cfg.resume  = cfg.resume.lower()  in ('true', '1', 'yes')
    cfg.dry_run = cfg.dry_run.lower() in ('true', '1', 'yes')

    log_dir      = os.path.join(cfg.result_dir, 'logs')
    summary_path = os.path.join(cfg.result_dir,
                                'run_all_physionet2020_sleep_v7_summary.csv')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ── Build queue ───────────────────────────────────────────────────────────
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
    print('\n' + '=' * 76)
    print(' Physionet2020 Sleep-CL V7 Batch')
    print(' (BN-safe state_dict rollback + scenario-aware threshold)')
    print(' Scenarios : {}'.format(cfg.scenarios))
    print(' Groups    : {}'.format(cfg.groups))
    print(' Agents    : {}'.format(cfg.agents))
    print(' Queue     : {} experiments'.format(total))
    print(' Result dir: {}'.format(cfg.result_dir))
    print('-' * 76)
    print(' V7 changes vs V6:')
    print('  [BN FIX]  Rollback uses state_dict() — BN running stats restored.')
    print('    V6 bug: named_parameters() missed 9 BN buffers.')
    print('    ASER Domain-IL seed 1 dropped to 24% in V6 — BN shift cause.')
    print('  [ROLLBACK THRESHOLD per scenario]:')
    print('    Class-IL  : 0.70 (완화 — was too conservative)')
    print('    Time-IL   : 0.80 (유지)')
    print('    Domain-IL : 0.90 (강화 — BN shift risk)')
    print('  [rollback_n_batches = 20] more reliable accuracy proxy')
    print('  [aser_rollback_factor = {:.2f}] ASER hard-example buffer correction'.format(
        cfg.aser_rollback_factor))
    print('  [S1 selective] LwF+SI in Class-IL → no-merge (inherited from V6)')
    print('-' * 76)
    for sc in cfg.scenarios:
        hp = SCENARIO_HPARAMS_V7.get(sc, {})
        print('  {:<10}  rollback_thr={:.2f}  n_batches={:2d}  '
              'merge_lam={:.3f}  every_n={:d}  min_fgt={:.1f}%'.format(
                  sc,
                  hp.get('rollback_threshold',  0.85),
                  hp.get('rollback_n_batches',    20),
                  hp.get('merge_lam',           0.15),
                  hp.get('sleep_every_n_tasks',     1),
                  hp.get('min_fgt_to_sleep',      0.0)))
    print('=' * 76 + '\n')

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
        sym = {'ok': 'v', 'error': 'x', 'exception': '!',
               'skipped': '-'}.get(res['status'], '?')
        print('  {} {:>8.1f}s  (total: {:.0f}s, ~{:.0f}s left)'.format(
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

    print('\n' + '=' * 76)
    print(' Done in {:.0f}s  |  OK: {}  |  Failed: {}'.format(
        total_time, ok_count, err_count))
    print(' CSV: {}'.format(summary_path))
    print('=' * 76 + '\n')

    if err_count:
        print('Failed experiments:')
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
