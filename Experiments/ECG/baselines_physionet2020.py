import argparse
import json
import math
import os
import sys
import time

import numpy as np

_CLOPS_DIR = os.path.dirname(os.path.abspath(__file__))
_TSCIL_DIR = os.path.normpath(os.path.join(_CLOPS_DIR, '..', 'TSCIL'))
for _p in (_CLOPS_DIR, _TSCIL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─── CLOPS data / model ──────────────────────────────────────────────────────
from clops_backbone_model            import CLOPSBackboneModel
from clops_task_stream               import CLOPSTaskStream
from clops_task_stream_shared_label  import CLOPSSharedLabelTaskStream

# ─── Sleep-CL v6 mixin (V7 strategy: BN-safe rollback) ───────────────────────
from clops_sleep_mixin_v6 import make_clops_sleep_agent_v6

# ─── TSCIL agents ────────────────────────────────────────────────────────────
from agents.utils.name_match import agents as tscil_agents, _BASE_AGENTS_FOR_SLEEP
from utils.metrics           import compute_performance
from utils.utils             import seed_fixer, save_pickle, boolean_string, check_ram_usage
import utils.setup_elements  as _se

# ── Agent patches ─────────────────────────────────────────────────────────────
tscil_agents['sleep-CL'] = make_clops_sleep_agent_v6

from clops_lwf import CLOPSLwF
from clops_mas import CLOPSMas
from clops_si  import CLOPSSI

tscil_agents['LwF']              = CLOPSLwF
tscil_agents['MAS']              = CLOPSMas
tscil_agents['SI']               = CLOPSSI
_BASE_AGENTS_FOR_SLEEP['LwF']    = CLOPSLwF
_BASE_AGENTS_FOR_SLEEP['MAS']    = CLOPSMas
_BASE_AGENTS_FOR_SLEEP['SI']     = CLOPSSI

_SHARED_LABEL_SCENARIOS = ('Time-IL', 'Domain-IL')


# ─────────────────────────────────────────────────────────────────────────────
# TSCIL setup_elements patch
# ─────────────────────────────────────────────────────────────────────────────

def _patch_setup_elements(data_key, input_shape, n_total_tasks,
                          n_total_classes, n_cls_per_task, n_smp_per_cls_est):
    T, C = input_shape
    _se.input_size_match[data_key]   = [T, C]
    _se.n_tasks[data_key]            = n_total_tasks
    _se.n_tasks_val[data_key]        = max(1, n_total_tasks // 3)
    _se.n_tasks_exp[data_key]        = n_total_tasks
    _se.n_classes[data_key]          = n_total_classes
    _se.n_classes_per_task[data_key] = n_cls_per_task
    _se.n_smp_per_cls[data_key]      = n_smp_per_cls_est
    _se.data_path[data_key]          = ''


# ─────────────────────────────────────────────────────────────────────────────
# Task stream factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_task_stream(cfg):
    if cfg.cl_scenario == 'Class-IL':
        return CLOPSTaskStream(
            basepath           = cfg.basepath,
            dataset_name       = cfg.dataset,
            cl_scenario        = cfg.cl_scenario,
            fraction           = cfg.fraction,
            order              = cfg.order,
            val_split          = cfg.val_split,
            seed               = cfg.seed,
            basepath_physionet = cfg.basepath_physionet,
        ).setup()
    else:
        return CLOPSSharedLabelTaskStream(
            basepath           = cfg.basepath,
            dataset_name       = cfg.dataset,
            cl_scenario        = cfg.cl_scenario,
            fraction           = cfg.fraction,
            order              = cfg.order,
            val_split          = cfg.val_split,
            seed               = cfg.seed,
            basepath_physionet = cfg.basepath_physionet,
        ).setup()


# ─────────────────────────────────────────────────────────────────────────────
# Buffer size
# ─────────────────────────────────────────────────────────────────────────────

def _calc_mem_size(task_stream, cl_scenario: str,
                   mem_budget: float, max_mem_size: int) -> int:
    x0, y0    = task_stream.tasks[0][0]
    n_train_0 = x0.shape[0]
    n_cls_0   = len(set(y0.tolist()))
    spc       = max(1, n_train_0 // n_cls_0)

    if cl_scenario in _SHARED_LABEL_SCENARIOS:
        raw = int(mem_budget * spc) * task_stream.n_classes * task_stream.n_tasks
    else:
        raw = int(mem_budget * spc) * task_stream.n_classes

    mem_size = max(10, min(raw, max_mem_size))
    if raw > max_mem_size:
        print('[Buffer] mem_size {} capped to {}.'.format(raw, max_mem_size))
    return mem_size


# ─────────────────────────────────────────────────────────────────────────────
# Agent args
# ─────────────────────────────────────────────────────────────────────────────

def _build_agent_args(agent_name, data_key, mem_size, n_total_tasks,
                      n_total_classes, exp_path, run_id, cfg):
    return argparse.Namespace(
        agent           = agent_name,
        data            = data_key,
        run_id          = run_id,
        runs            = cfg.runs,
        exp_path        = exp_path,
        seed            = cfg.seed,
        epochs          = cfg.epochs,
        batch_size      = cfg.batch_size,
        lr              = cfg.lr,
        weight_decay    = 0.0,
        dropout         = cfg.dropout,
        lradj           = 'step15',
        early_stop      = cfg.early_stop,
        patience        = cfg.patience,
        scenario        = 'class',
        stream_split    = 'all',
        mem_size        = mem_size,
        mem_budget      = cfg.mem_budget,
        eps_mem_batch   = cfg.batch_size,
        retrieve        = 'random',
        update          = 'random',
        buffer_tracker  = False,
        er_mode         = cfg.er_mode,
        num_tasks       = n_total_tasks,
        n_total_classes = n_total_classes,
        head            = 'Linear',
        encoder         = 'CNN',
        norm            = 'BN',
        input_norm      = 'none',
        feature_dim     = 100,
        n_layers        = 3,
        device          = cfg.device,
        verbose         = cfg.verbose,
        tune            = False,
        tsne            = False,
        tsne_g          = False,
        cf_matrix       = False,
        criterion       = 'CE',
        ncm_classifier  = False,
        fix_order       = False,
        lambda_impt     = cfg.lambda_impt,
        ewc_mode        = 'separate',
        teacher_eval    = False,
        lambda_kd_lwf   = 1.0,
        lambda_kd_fmap  = 1e-2,
        fmap_kd_metric  = 'dtw',
        lambda_protoAug = 100.0,
        adaptive_weight = False,
        der_plus        = False,
        aser_k          = 3,
        aser_type       = 'asvm',
        aser_n_smp_cls  = 4,
        mc_retrieve     = cfg.mc_retrieve,
        beta_lr         = cfg.beta_lr,
        lambda_beta     = cfg.lambda_beta,
        epochs_g        = cfg.epochs_g,
        lr_g            = cfg.lr_g,
        recon_wt        = 0.1,
        mnemonics_epochs = 1,
        mnemonics_lr    = 1e-5,
        start_noise     = True,
        save_mode       = 0,
        n_samples_to_plot = 5,
        augment_batch   = False,
        visual_syn_feat = True,
        iterations_per_layer = 100,
        inversion_lr    = 1e-2,
        inchannel_scale = 10.0,
        xchannel_scale  = 1.0,
        feat_scale      = 1.0,
        k_freq          = -1,
        regularize_freq_on_feat = False,
        er_sub_type     = 'part',
        # ── CL scenario ───────────────────────────────────────────────────
        cl_scenario     = cfg.cl_scenario,
        # ── sleep-CL v2 params ────────────────────────────────────────────
        sleep_base_agent        = getattr(cfg, 'sleep_base_agent',        'ER'),
        sleep_steps             = getattr(cfg, 'sleep_steps',             100),
        sleep_lr                = getattr(cfg, 'sleep_lr',                1e-3),
        down_alpha              = getattr(cfg, 'down_alpha',              0.85),
        merge_lam               = getattr(cfg, 'merge_lam',               0.5),
        lambda_distill          = getattr(cfg, 'lambda_distill',          0.5),
        lambda_div              = getattr(cfg, 'lambda_div',              0.2),
        lambda_old              = getattr(cfg, 'lambda_old',              1.0),
        fisher_protect_quantile = getattr(cfg, 'fisher_protect_quantile', 0.7),
        alpha_protect           = getattr(cfg, 'alpha_protect',           0.95),
        focused_steps_per_task  = getattr(cfg, 'focused_steps_per_task',  10),
        sleep_samples_per_class = getattr(cfg, 'sleep_samples_per_class', 20),
        replace_bn_with_gn      = getattr(cfg, 'replace_bn_with_gn',     False),
        sleep_steps_min         = getattr(cfg, 'sleep_steps_min',         50),
        sleep_steps_max         = getattr(cfg, 'sleep_steps_max',        200),
        lambda_distill_min      = getattr(cfg, 'lambda_distill_min',     -1.0),
        lambda_distill_max      = getattr(cfg, 'lambda_distill_max',     -1.0),
        spindle_base_steps_min  = getattr(cfg, 'spindle_base_steps_min',   2),
        spindle_base_steps_max  = getattr(cfg, 'spindle_base_steps_max',  10),
        swr_steps               = getattr(cfg, 'swr_steps',
                                          getattr(cfg, 'focused_steps_per_task', 10)),
        skip_phase1             = getattr(cfg, 'skip_phase1',             False),
        skip_phase2a            = getattr(cfg, 'skip_phase2a',            False),
        skip_phase2b            = getattr(cfg, 'skip_phase2b',            False),
        skip_phase3             = getattr(cfg, 'skip_phase3',             False),
        adaptive_merge          = getattr(cfg, 'adaptive_merge',          True),
        use_oscillation_structure = getattr(cfg, 'use_oscillation_structure', True),
        skip_phase1_threshold   = getattr(cfg, 'skip_phase1_threshold',  0.995),
        merge_decay             = getattr(cfg, 'merge_decay',             0.10),
        merge_lam_floor         = getattr(cfg, 'merge_lam_floor',         0.05),
        # ── sleep-CL v3 params ────────────────────────────────────────────
        swr_only                = getattr(cfg, 'swr_only',               False),
        auto_swr_only           = getattr(cfg, 'auto_swr_only',          True),
        sleep_every_n_tasks     = getattr(cfg, 'sleep_every_n_tasks',    1),
        min_fgt_to_sleep        = getattr(cfg, 'min_fgt_to_sleep',       0.0),
        # ── sleep-CL v7 params ────────────────────────────────────────────
        use_rollback            = getattr(cfg, 'use_rollback',           True),
        rollback_threshold      = getattr(cfg, 'rollback_threshold',     0.85),
        rollback_n_batches      = getattr(cfg, 'rollback_n_batches',     20),
        aser_rollback_factor    = getattr(cfg, 'aser_rollback_factor',   0.75),
        # Fisher merge: off by default (retained as opt-in)
        use_fisher_merge        = getattr(cfg, 'use_fisher_merge',       False),
        fisher_beta             = getattr(cfg, 'fisher_beta',            1.0),
        fisher_n_batches        = getattr(cfg, 'fisher_n_batches',       10),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single CL run
# ─────────────────────────────────────────────────────────────────────────────

def _run_once(task_stream, agent_name, agent_args,
              n_initial_classes, n_channels, dropout, device,
              fixed_head: bool):
    model = CLOPSBackboneModel(
        n_initial_classes = n_initial_classes,
        n_channels        = n_channels,
        dropout_type      = 'drop1d',
        p1 = dropout, p2 = dropout, p3 = dropout,
    ).to(device)

    if fixed_head:
        model.update_head = lambda n_new=None, task_now=None: None

    AgentClass = tscil_agents[agent_name]
    agent      = AgentClass(model=model, args=agent_args)

    for task in task_stream.tasks:
        agent.learn_task(task)
        agent.evaluate(task_stream)

    sleep_log = []
    if hasattr(agent, '_sleep_plasticity_log') and agent._sleep_plasticity_log:
        sleep_log = agent._sleep_plasticity_log

    return agent.Acc_tasks, sleep_log


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Physionet2020 v7 runner (sleep-CL v6: BN-safe state_dict rollback)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Dataset / scenario ────────────────────────────────────────────────────
    parser.add_argument('--dataset',    type=str, default='physionet2020',
                        choices=['physionet2020', 'chapman'])
    parser.add_argument('--cl_scenario', type=str, default='Class-IL',
                        choices=['Class-IL', 'Time-IL', 'Domain-IL'])
    parser.add_argument('--basepath',   type=str,
                        default='/mnt/CONIRepo/ykkim/SLCL/datasets/')
    parser.add_argument('--basepath_physionet', type=str,
                        default='/mnt/CONIRepo/ykkim/SLCL/datasets/Physionet2020/patient_data/')
    parser.add_argument('--fraction',   type=float, default=0.9)
    parser.add_argument('--order',      type=int,   default=2)
    parser.add_argument('--val_split',  type=float, default=0.1)

    # ── Agent ─────────────────────────────────────────────────────────────────
    parser.add_argument('--agent', type=str, default='ER',
                        choices=['SFT', 'ER', 'EWC', 'LwF', 'SI', 'MAS', 'DT2W',
                                 'ASER', 'Herding', 'CLOPS', 'DER', 'GR',
                                 'FastICARL', 'sleep-CL',
                                 'SRC', 'WSCL', 'SIESTA'])

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument('--runs',         type=int,   default=3)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--dropout',      type=float, default=0.0)
    parser.add_argument('--er_mode',      type=str,   default='task',
                        choices=['task', 'online'])
    parser.add_argument('--mem_budget',   type=float, default=0.05)
    parser.add_argument('--max_mem_size', type=int,   default=1500)
    parser.add_argument('--early_stop',   type=boolean_string, default=True)
    parser.add_argument('--patience',     type=int,   default=20)
    parser.add_argument('--lambda_impt',  type=float, default=10000.0)
    parser.add_argument('--mc_retrieve',  type=boolean_string, default=False)
    parser.add_argument('--beta_lr',      type=float, default=1e-4)
    parser.add_argument('--lambda_beta',  type=float, default=1.0)
    parser.add_argument('--epochs_g',     type=float, default=200)
    parser.add_argument('--lr_g',         type=float, default=1e-3)

    # ── sleep-CL v2/v3 params ─────────────────────────────────────────────────
    parser.add_argument('--sleep_base_agent', type=str, default='ER',
                        choices=['SFT', 'ER', 'EWC', 'LwF', 'SI', 'MAS', 'DT2W',
                                 'ASER', 'Herding', 'CLOPS', 'DER', 'GR', 'FastICARL'])
    parser.add_argument('--sleep_steps',    type=int,   default=150)
    parser.add_argument('--sleep_lr',       type=float, default=1e-3)
    parser.add_argument('--down_alpha',     type=float, default=0.90)
    parser.add_argument('--merge_lam',      type=float, default=0.15)
    parser.add_argument('--lambda_distill', type=float, default=0.0)
    parser.add_argument('--lambda_div',     type=float, default=0.10)
    parser.add_argument('--lambda_old',     type=float, default=1.0)
    parser.add_argument('--fisher_protect_quantile', type=float, default=0.80)
    parser.add_argument('--alpha_protect',  type=float, default=0.97)
    parser.add_argument('--focused_steps_per_task',  type=int, default=10)
    parser.add_argument('--sleep_samples_per_class', type=int, default=50)
    # ── WSCL params ───────────────────────────────────────────────────────────
    parser.add_argument('--wscl_sleep_epochs', type=int,   default=3)
    parser.add_argument('--wscl_lambda_ewc',   type=float, default=1000.0)
    parser.add_argument('--wscl_sleep_lr',     type=float, default=0.0)
    # ── SIESTA params ─────────────────────────────────────────────────────────
    parser.add_argument('--siesta_sleep_epochs', type=int,   default=5)
    parser.add_argument('--siesta_sleep_lr',     type=float, default=0.0)
    parser.add_argument('--siesta_sleep_bs',     type=int,   default=0)
    parser.add_argument('--replace_bn_with_gn', type=boolean_string, default=False)
    parser.add_argument('--sleep_steps_min',        type=int,   default=100)
    parser.add_argument('--sleep_steps_max',        type=int,   default=200)
    parser.add_argument('--lambda_distill_min',     type=float, default=-1.0)
    parser.add_argument('--lambda_distill_max',     type=float, default=-1.0)
    parser.add_argument('--spindle_base_steps_min', type=int,   default=5)
    parser.add_argument('--spindle_base_steps_max', type=int,   default=15)
    parser.add_argument('--skip_phase1',  type=boolean_string, default=False)
    parser.add_argument('--skip_phase2a', type=boolean_string, default=False)
    parser.add_argument('--skip_phase2b', type=boolean_string, default=False)
    parser.add_argument('--skip_phase3',  type=boolean_string, default=False)
    parser.add_argument('--adaptive_merge', type=boolean_string, default=True)
    parser.add_argument('--use_oscillation_structure', type=boolean_string, default=True)
    parser.add_argument('--skip_phase1_threshold', type=float, default=0.995)
    parser.add_argument('--merge_decay',    type=float, default=0.00)
    parser.add_argument('--merge_lam_floor', type=float, default=0.10)
    parser.add_argument('--swr_only', type=boolean_string, default=False)
    parser.add_argument('--auto_swr_only', type=boolean_string, default=True)
    parser.add_argument('--sleep_every_n_tasks', type=int, default=1)
    parser.add_argument('--min_fgt_to_sleep', type=float, default=0.0)

    # ── sleep-CL v7 params ────────────────────────────────────────────────────
    parser.add_argument('--use_rollback', type=boolean_string, default=True,
                        help='Strategy 3: rollback to θ_wake if post-sleep acc drops')
    parser.add_argument('--rollback_threshold', type=float, default=0.85,
                        help='rollback if post_acc < threshold × pre_acc'
                             ' (set per-scenario: Class-IL=0.70, Time-IL=0.80, Domain-IL=0.90)')
    parser.add_argument('--rollback_n_batches', type=int, default=20,
                        help='Batches for pre/post accuracy proxy (more = more reliable)')
    parser.add_argument('--aser_rollback_factor', type=float, default=0.75,
                        help='Scales rollback_threshold for ASER (hard-example buffer'
                             ' gives lower proxy acc; factor<1 makes rollback less strict)')
    # Fisher merge: opt-in only
    parser.add_argument('--use_fisher_merge', type=boolean_string, default=False)
    parser.add_argument('--fisher_beta', type=float, default=1.0)
    parser.add_argument('--fisher_n_batches', type=int, default=10)

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--seed',       type=int,            default=1234)
    parser.add_argument('--device',     type=str,            default='cuda')
    parser.add_argument('--verbose',    type=boolean_string, default=True)
    parser.add_argument('--result_dir', type=str,
                        default='./tscil_results/physionet2020_v7')
    parser.add_argument('--debug',      type=boolean_string, default=False)

    cfg = parser.parse_args()
    cfg.device = 'cuda'

    is_shared_label = cfg.cl_scenario in _SHARED_LABEL_SCENARIOS
    label_mode      = 'shared' if is_shared_label else 'offset'

    # ── Result path ───────────────────────────────────────────────────────────
    scenario_tag = cfg.cl_scenario.replace('-', '')
    if cfg.agent == 'sleep-CL':
        exp_name = 'v7_sleepCL_{}_{}_{}'.format(
            cfg.sleep_base_agent, cfg.dataset, scenario_tag)
    else:
        exp_name = 'v7_{}_{}_{}'.format(cfg.agent, cfg.dataset, scenario_tag)

    result_root = os.path.join(cfg.result_dir, 'debug' if cfg.debug else 'exp')
    exp_path    = os.path.join(result_root, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2)

    print('\n========================================')
    print(' Agent      :', cfg.agent)
    print(' Dataset    :', cfg.dataset)
    print(' Scenario   :', cfg.cl_scenario, '({})'.format(label_mode))
    if cfg.agent == 'sleep-CL':
        print(' Sleep base :', cfg.sleep_base_agent)
        print(' Fix BN     : state_dict rollback (params + BN buffers)')
        print(' S1 selective: LwF/SI in Class-IL → no-merge')
        print(' Rollback   : use={}, thr={:.2f}, n_batches={}'.format(
            cfg.use_rollback, cfg.rollback_threshold, cfg.rollback_n_batches))
        print(' ASER factor: {:.2f} (effective_thr for ASER={:.2f})'.format(
            cfg.aser_rollback_factor,
            cfg.rollback_threshold * cfg.aser_rollback_factor))
        print(' Alt5 swr_only  :', cfg.swr_only,
              '(auto={})'.format(cfg.auto_swr_only))
        print(' Alt6 min_fgt   : {:.1f}%'.format(cfg.min_fgt_to_sleep))
        print(' Alt7 every_n   :', cfg.sleep_every_n_tasks)
        print(' Alt1 alpha     :', cfg.down_alpha, '(adaptive)')
        print(' Alt4 merge_lam :', cfg.merge_lam,
              '(decay={}, floor={})'.format(cfg.merge_decay, cfg.merge_lam_floor))
    print(' Device     :', cfg.device)
    print(' Result dir :', exp_path)
    print('========================================\n')

    # ── Task stream ───────────────────────────────────────────────────────────
    task_stream = _build_task_stream(cfg)

    n_total_tasks = task_stream.n_tasks
    n_total_cls   = task_stream.n_classes
    T, C          = task_stream.input_shape

    if is_shared_label:
        n_initial_cls  = n_total_cls
        n_cls_per_task = n_total_cls
    else:
        n_cls_per_task = len(set(task_stream.tasks[0][0][1].tolist()))
        n_initial_cls  = n_cls_per_task

    # ── Buffer size ───────────────────────────────────────────────────────────
    mem_size = _calc_mem_size(task_stream, cfg.cl_scenario,
                              cfg.mem_budget, cfg.max_mem_size)
    buf_mb   = mem_size * T * C * 4 / 1024 / 1024
    print('Buffer: {} samples ({:.1f} MiB) | label_mode={} | n_tasks={}'.format(
        mem_size, buf_mb, label_mode, n_total_tasks))

    # ── Register in TSCIL setup_elements ─────────────────────────────────────
    denominator   = cfg.mem_budget * n_cls_per_task * n_total_tasks
    n_smp_for_buf = max(1, math.ceil(mem_size / max(denominator, 1e-9)))
    data_key      = 'clops_v7_{}_{}_{}'.format(cfg.dataset, scenario_tag, T)

    _patch_setup_elements(
        data_key          = data_key,
        input_shape       = (T, C),
        n_total_tasks     = n_total_tasks,
        n_total_classes   = n_total_cls,
        n_cls_per_task    = n_cls_per_task,
        n_smp_per_cls_est = n_smp_for_buf,
    )

    # ── Multiple runs ─────────────────────────────────────────────────────────
    Acc_valid_runs        = []
    Acc_test_runs         = []
    sleep_plasticity_runs = []
    t_start               = time.time()

    for run in range(cfg.runs):
        seed_fixer(cfg.seed + run)
        print('\n------ Run {}/{} ------'.format(run + 1, cfg.runs))

        run_path = os.path.join(exp_path, 'run_{}'.format(run))
        os.makedirs(run_path, exist_ok=True)

        agent_args = _build_agent_args(
            agent_name      = cfg.agent,
            data_key        = data_key,
            mem_size        = mem_size,
            n_total_tasks   = n_total_tasks,
            n_total_classes = n_total_cls,
            exp_path        = run_path,
            run_id          = run,
            cfg             = cfg,
        )

        acc_tasks, sleep_log = _run_once(
            task_stream       = task_stream,
            agent_name        = cfg.agent,
            agent_args        = agent_args,
            n_initial_classes = n_initial_cls,
            n_channels        = C,
            dropout           = cfg.dropout,
            device            = cfg.device,
            fixed_head        = is_shared_label,
        )

        Acc_valid_runs.append(acc_tasks['valid'])
        Acc_test_runs.append(acc_tasks['test'])
        if sleep_log:
            sleep_plasticity_runs.append(sleep_log)
        print('Run {} done  ({:.1f}s elapsed)'.format(
            run + 1, time.time() - t_start))

    # ── Aggregate ─────────────────────────────────────────────────────────────
    Acc_valid_runs = np.array(Acc_valid_runs)
    Acc_test_runs  = np.array(Acc_test_runs)

    for split_name, Acc_arr in [('Valid', Acc_valid_runs), ('Test', Acc_test_runs)]:
        avg_end_acc, avg_end_fgt, avg_cur_acc, avg_acc, avg_bwtp = \
            compute_performance(Acc_arr)
        print('\n[{}] End_Acc={:.2f}  End_Fgt={:.2f}  Cur_Acc={:.2f}'
              '  Avg_Acc={:.2f}  BWT+={:.2f}'.format(
                  split_name,
                  np.around(avg_end_acc[0], 2),
                  np.around(avg_end_fgt[0], 2),
                  np.around(avg_cur_acc[0], 2),
                  np.around(avg_acc[0][-1], 2),
                  np.around(avg_bwtp[0],    2)))

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        'time'            : time.time() - t_start,
        'acc_array_val'   : Acc_valid_runs,
        'acc_array_test'  : Acc_test_runs,
        'ram'             : check_ram_usage(),
        'agent'           : cfg.agent,
        'dataset'         : cfg.dataset,
        'cl_scenario'     : cfg.cl_scenario,
        'label_mode'      : label_mode,
        'n_classes'       : n_total_cls,
        'config'          : vars(cfg),
    }
    if sleep_plasticity_runs:
        result['sleep_plasticity'] = sleep_plasticity_runs

    save_pickle(result, os.path.join(exp_path, 'result.pkl'))
    print('\nResults saved to:', exp_path)


if __name__ == '__main__':
    main()
