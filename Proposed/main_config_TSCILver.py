import argparse
import sys
import os
import json
import time
import torch
import agents.utils.name_match as _name_match_module
from experiment.exp import experiment_multiple_runs
from utils.utils import Logger, boolean_string


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train the continual learning agent on task sequence (v2: improved sleep-CL)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # #################### Main setting ####################
    parser.add_argument('--agent', dest='agent', default='sleep-CL', type=str,
                        choices=['SFT', 'Offline',
                                 'LwF', 'EWC', 'SI', 'MAS', 'DT2W',
                                 'ER', 'ASER', 'DER', 'Herding', 'CLOPS', 'ER_Sub',
                                 'Mnemonics', 'Inversion', 'GR',
                                 'FastICARL', 'sleep-CL', 'SRC',
                                 'WSCL', 'SIESTA'],
                        help='Continual learning agent')

    parser.add_argument('--scenario', type=str, default='class',
                        choices=['class', 'domain'])

    parser.add_argument('--stream_split', type=str, default='exp',
                        choices=['val', 'exp', 'all'])

    parser.add_argument('--data', dest='data', default=['all'], type=str,
                        nargs='+',
                        choices=['all',
                                 'har', 'uwave', 'dailysports', 'wisdm',
                                 'sines'],
                        help='Dataset(s) to run. Pass a single name, multiple names, or "all".')

    # Backbone
    parser.add_argument('--encoder', dest='encoder', default='CNN', type=str,
                        choices=['CNN'])

    # Classifier
    parser.add_argument('--head', dest='head', default='Linear', type=str,
                        choices=['Linear'])
    parser.add_argument('--criterion', dest='criterion', default='CE', type=str,
                        choices=['CE'])
    parser.add_argument('--ncm_classifier', dest='ncm_classifier', type=boolean_string, default=False)

    # Normalisation layers
    parser.add_argument('--norm', dest='norm', default='BN', type=str,
                        choices=['BN', 'LN'])
    parser.add_argument('--input_norm', dest='input_norm', default='IN', type=str,
                        choices=['LN', 'IN', 'ZScore', 'none'])

    """ General params """
    parser.add_argument('--runs',        dest='runs',        default=5,     type=int)
    parser.add_argument('--epochs',      dest='epochs',      default=100,   type=int)
    parser.add_argument('--batch_size',  dest='batch_size',  default=32,    type=int)
    parser.add_argument('--lr',          dest='lr',          default=1e-3,  type=float)
    parser.add_argument('--lradj',       type=str,           default='step15')
    parser.add_argument('--early_stop',  type=boolean_string, default=True)
    parser.add_argument('--patience',    type=int,           default=20)
    parser.add_argument('--weight_decay',dest='weight_decay', default=0,    type=float)
    parser.add_argument('--dropout',     dest='dropout',     default=0,     type=float)
    parser.add_argument('--feature_dim', dest='feature_dim', default=128,   type=int)
    parser.add_argument('--n_layers',    dest='n_layers',    default=4,     type=int)

    # Nuisance variables
    parser.add_argument('--tune',       type=boolean_string, default=False)
    parser.add_argument('--debug',      type=boolean_string, default=True)
    parser.add_argument('--seed',       dest='seed',         default=1234,  type=int)
    parser.add_argument('--device',     dest='device',       default='cpu', type=str)
    parser.add_argument('--verbose',    type=boolean_string, default=True)
    parser.add_argument('--exp_start_time', dest='exp_start_time', type=str)
    parser.add_argument('--fix_order',  type=boolean_string, default=False)
    parser.add_argument('--cf_matrix',  type=boolean_string, default=True)
    parser.add_argument('--tsne',       type=boolean_string, default=False)
    parser.add_argument('--tsne_g',     type=boolean_string, default=True)

    # ######################## Methods-related params ###########################
    # Experience Replay
    parser.add_argument('--er_mode',       type=str,   default='task', choices=['online', 'task'])
    parser.add_argument('--mem_budget',    type=float, default=0.05)
    parser.add_argument('--buffer_tracker',type=boolean_string, default=False)
    parser.add_argument('--der_plus',      type=boolean_string, default=False)
    parser.add_argument('--er_sub_type',   type=str,   default='part', choices=['balanced', 'part'])

    # KD: LwF / DT2W
    parser.add_argument('--teacher_eval',     type=boolean_string, default=False)
    parser.add_argument('--lambda_kd_lwf',    dest='lambda_kd_lwf',    default=1,    type=float)
    parser.add_argument('--lambda_kd_fmap',   dest='lambda_kd_fmap',   default=1e-2, type=float)
    parser.add_argument('--fmap_kd_metric',   dest='fmap_kd_metric',   default='dtw', type=str,
                        choices=['dtw', 'euclidean', 'pod_temporal', 'pod_variate'])
    parser.add_argument('--lambda_protoAug',  dest='lambda_protoAug',  default=100,  type=float)
    parser.add_argument('--adaptive_weight',  type=boolean_string, default=False)

    # EWC / MAS / SI
    parser.add_argument('--lambda_impt', dest='lambda_impt', default=10000, type=float)
    parser.add_argument('--ewc_mode',    dest='ewc_mode',    default='separate', type=str,
                        choices=['separate', 'online'])

    # ASER
    parser.add_argument('--aser_k',         dest='aser_k',         default=3,      type=int)
    parser.add_argument('--aser_type',       dest='aser_type',       default='asvm', type=str,
                        choices=['neg_sv', 'asv', 'asvm'])
    parser.add_argument('--aser_n_smp_cls',  dest='aser_n_smp_cls',  default=4,      type=float)

    # CLOPS
    parser.add_argument('--mc_retrieve',  type=boolean_string, default=False)
    parser.add_argument('--beta_lr',      dest='beta_lr',      default=1e-4, type=float)
    parser.add_argument('--lambda_beta',  dest='lambda_beta',  default=1,    type=float)

    # Generative Replay
    parser.add_argument('--epochs_g',  type=float, default=500)
    parser.add_argument('--lr_g',      type=float, default=1e-3)
    parser.add_argument('--recon_wt',  type=float, default=0.1)

    # Mnemonics
    parser.add_argument('--mnemonics_epochs', default=1,    type=int)
    parser.add_argument('--mnemonics_lr',     type=float,   default=1e-5)

    # #################### sleep-CL ####################
    parser.add_argument('--sleep_base_agent', type=str, default=['ASER'],
                        nargs='+',
                        choices=['all', 'ER', 'EWC', 'LwF', 'SI', 'MAS', 'CLOPS', 'GR', 'DER'],
                        help='Base CL agent(s) that sleep-CL wraps.')
    parser.add_argument('--sleep_steps',     type=int,   default=100,
                        help='Number of sleep-phase optimisation steps per task')
    parser.add_argument('--sleep_lr',        type=float, default=1e-3,
                        help='Learning rate for sleep-phase optimiser')
    parser.add_argument('--down_alpha',      type=float, default=0.85,
                        help='Downregulation factor α ∈ (0,1): W ← α·W before sleep')
    parser.add_argument('--merge_lam',       type=float, default=0.5,
                        help='Merge lambda λ: θ ← (1-λ)·θ_wake + λ·θ_sleep')
    parser.add_argument('--lambda_distill',  type=float, default=0.5,
                        help='Weight of KL-distillation loss in sleep objective')
    parser.add_argument('--lambda_div',      type=float, default=0.2,
                        help='Weight of diversity (feature covariance) loss in sleep objective')
    parser.add_argument('--lambda_old',      type=float, default=1.0,
                        help='Weight of old-knowledge distillation loss (KL from pre-wake model)')
    parser.add_argument('--lambda_importance', type=float, default=0.5,
                        help='Weight of importance regularisation (EWC/MAS/SI/DER)')
    parser.add_argument('--fisher_protect_quantile', type=float, default=0.7,
                        help='Fisher importance quantile for two-tier downregulation')
    parser.add_argument('--alpha_protect',   type=float, default=0.95,
                        help='Mild downreg scale for high-Fisher weights')
    parser.add_argument('--sleep_samples_per_class', type=int, default=20,
                        help='Samples per class stored in the internal sleep memory')

    parser.add_argument('--downreg_stabilize_steps', type=int, default=10,
                        help='Phase 1.5: steps of L_cons-only optimisation after downreg '
                             '(distill_scale=0, old_scale=0) to stabilise before teacher refresh')
    parser.add_argument('--use_post_downreg_teacher', type=boolean_string, default=True,
                        help='Method 2: snapshot teacher_wake from post-downreg model '
                             '(prevents L_dist from reversing Phase 1 downreg)')

    parser.add_argument('--use_fisher_downreg',        type=boolean_string, default=True)
    parser.add_argument('--use_dormant_reactivation',  type=boolean_string, default=True)
    parser.add_argument('--use_oscillation_structure', type=boolean_string, default=True)
    parser.add_argument('--use_grad_projection',       type=boolean_string, default=True)
    parser.add_argument('--adaptive_merge',            type=boolean_string, default=True)
    parser.add_argument('--adaptive_lr',               type=boolean_string, default=True)
    parser.add_argument('--adaptive_sleep_steps',      type=boolean_string, default=True)
    parser.add_argument('--adaptive_spindle',          type=boolean_string, default=True)
    parser.add_argument('--adaptive_lambda_imp',       type=boolean_string, default=True)

    parser.add_argument('--skip_phase1',  type=boolean_string, default=False,
                        help='Skip Phase 1 (downreg + dormant + Phase1.5 + Method2)')
    parser.add_argument('--skip_phase2a', type=boolean_string, default=False,
                        help='Skip Phase 2a (global flat consolidation loop)')
    parser.add_argument('--skip_phase2b', type=boolean_string, default=False,
                        help='Skip Phase 2b (spindle — age-weighted per-task)')
    parser.add_argument('--skip_phase3',  type=boolean_string, default=False,
                        help='Skip Phase 3 (SWR — boundary-sample intense replay)')

    # ── Architecture ────────────────────────────────────────────────────────
    parser.add_argument('--replace_bn_with_gn', type=boolean_string, default=True,
                        help='Replace BatchNorm with GroupNorm (num_groups=32) '
                             'before sleep. Necessary for stable sleep consolidation '
                             'with multi-task mixed-batch replay.')

    # #################### SRC ####################
    parser.add_argument('--src_decay',     type=float, default=0.9)
    parser.add_argument('--src_inc',       type=float, default=0.01)
    parser.add_argument('--src_dec',       type=float, default=0.001)
    parser.add_argument('--src_alpha',     type=float, default=1.0)
    parser.add_argument('--src_W_inh',     type=float, default=0.0)
    parser.add_argument('--src_beta',      type=float, default=1.0)
    parser.add_argument('--src_threshold', type=float, default=1.0)
    parser.add_argument('--src_t_ref',     type=int,   default=5)
    parser.add_argument('--src_normW',     type=boolean_string, default=True)
    parser.add_argument('--src_gamma',     type=float, default=1.0)

    # #################### WSCL ####################
    parser.add_argument('--wscl_sleep_epochs', type=int,   default=3,
                        help='NREM sleep: gradient epochs over balanced replay')
    parser.add_argument('--wscl_lambda_ewc',   type=float, default=1000.0,
                        help='NREM sleep: EWC regularisation strength')
    parser.add_argument('--wscl_sleep_lr',     type=float, default=0.0,
                        help='NREM sleep learning rate (0 = use --lr)')

    # #################### SIESTA ####################
    parser.add_argument('--siesta_sleep_epochs', type=int,   default=5,
                        help='SIESTA sleep: replay epochs after each task')
    parser.add_argument('--siesta_sleep_lr',     type=float, default=0.0,
                        help='SIESTA sleep peak LR for OneCycleLR (0 = use --lr)')
    parser.add_argument('--siesta_sleep_bs',     type=int,   default=0,
                        help='SIESTA sleep batch size (0 = use --batch_size)')

    # Model Inversion
    parser.add_argument('--start_noise',              default=True,  type=boolean_string)
    parser.add_argument('--save_mode',                default=0,     type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--n_samples_to_plot',        default=5,     type=int)
    parser.add_argument('--augment_batch',            default=False, type=boolean_string)
    parser.add_argument('--visual_syn_feat',          default=True,  type=boolean_string)
    parser.add_argument('--iterations_per_layer',     type=int,      default=100)
    parser.add_argument('--inversion_lr',             type=float,    default=1e-2)
    parser.add_argument('--inchannel_scale',          type=float,    default=10)
    parser.add_argument('--xchannel_scale',           type=float,    default=1)
    parser.add_argument('--feat_scale',               type=float,    default=1)
    parser.add_argument('--k_freq',                   type=int,      default=-1)
    parser.add_argument('--regularize_freq_on_feat',  default=False, type=boolean_string)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Resolve dataset list ────────────────────────────────────────────────
    _ALL_DATASETS = ['har', 'uwave', 'dailysports', 'grabmyo', 'wisdm', 'sines']
    req_data = args.data if isinstance(args.data, list) else [args.data]
    datasets_to_run = _ALL_DATASETS if 'all' in req_data else req_data

    # ── Resolve sleep_base_agent list ───────────────────────────────────────
    # _ALL_SLEEP_AGENTS = ['ER', 'EWC', 'LwF', 'SI', 'MAS', 'CLOPS', 'GR', 'DER']
    _ALL_SLEEP_AGENTS = ['DER']
    # Extend to ['ER', 'EWC', 'SI', 'MAS', 'LwF', 'GR'] when running full suite
    if args.agent == 'sleep-CL':
        req_agents = args.sleep_base_agent
        if isinstance(req_agents, str):
            req_agents = [req_agents]
        sleep_agents_to_run = _ALL_SLEEP_AGENTS if 'all' in req_agents else req_agents
    else:
        sleep_agents_to_run = [None]

    is_sleep_cl  = args.agent == 'sleep-CL'
    total_runs   = len(datasets_to_run) * len(sleep_agents_to_run)
    multi_data   = len(datasets_to_run) > 1
    multi_agent  = is_sleep_cl and len(sleep_agents_to_run) > 1

    exp_start_time = time.strftime("%b-%d-%H-%M-%S", time.localtime())
    exp_path_0 = './result/exp/' if not args.debug else './result/exp/debug'
    _orig_stdout = sys.stdout
    run_idx = 0

    for dataset in datasets_to_run:
        args.data = dataset

        for base_agent in sleep_agents_to_run:
            if is_sleep_cl and base_agent is not None:
                args.sleep_base_agent = base_agent
            run_idx += 1

            # Build result directory path
            exp_path_1 = args.encoder + '_' + dataset
            if is_sleep_cl and base_agent is not None:
                # 'v2' suffix distinguishes from main_config.py results
                exp_path_2 = f"{args.agent}_{base_agent}_v2_{args.norm}"
            else:
                exp_path_2 = args.agent + '_v2_' + args.norm

            exp_path = os.path.join(exp_path_0, exp_path_1, exp_path_2)
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)

            args.exp_path = exp_path
            log_dir = os.path.join(exp_path, 'log.txt')
            with open(log_dir, 'w') as f:
                json.dump(vars(args), f, indent=2)

            sys.stdout = Logger(log_dir)

            if multi_data or multi_agent:
                print(f"\n{'='*60}")
                label = (f"data={dataset}  |  base_agent={base_agent}"
                         if is_sleep_cl else f"data={dataset}  |  agent={args.agent}")
                print(f"  [{run_idx}/{total_runs}]  {label}")
                print(f"{'='*60}\n")
            print(args)

            try:
                experiment_multiple_runs(args)
            except Exception as exc:
                import traceback
                err_label = (f"base_agent={base_agent}"
                             if is_sleep_cl else f"agent={args.agent}")
                print(f"\n[ERROR] data={dataset} {err_label}: {exc}")
                traceback.print_exc()
            finally:
                sys.stdout = _orig_stdout
