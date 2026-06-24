from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import torch
import torch.nn.functional as F
import copy

from agents._sleep_opt import *
from metrics_abc import effective_rank, dormant_ratio, update_ratio
import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) Loss terms
# ============================================================

def loss_consolidation_ce(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """L_cons: supervised consolidation on memory buffer."""
    logits = forward_with_features(model, xb)[0]
    return F.cross_entropy(logits, yb)


def loss_distill_kl(
    student: nn.Module,
    teacher: nn.Module,
    x: torch.Tensor,
    T: float = 2.0,
    detach_teacher: bool = True
) -> torch.Tensor:
    """
    L_distill: KL divergence between teacher and student output distributions.
    Classic distillation loss:
      KL( softmax(z_t/T) || softmax(z_s/T) ) * T^2
    """
    z_s = forward_with_features(student, x)[0]
    with torch.no_grad() if detach_teacher else torch.enable_grad():
        z_t = forward_with_features(teacher, x)[0]

    p_t = F.softmax(z_t / max(T, 1e-6), dim=1)
    log_p_s = F.log_softmax(z_s / max(T, 1e-6), dim=1)

    # KL(p_t || p_s) = sum p_t (log p_t - log p_s)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean")
    return kl * (T * T)


def loss_distill_feature_l2(
    student: nn.Module,
    teacher: nn.Module,
    x: torch.Tensor,
    detach_teacher: bool = True
) -> torch.Tensor:
    """
    Optional: feature-level distillation to preserve representation geometry.
    """
    _, hs = forward_with_features(student, x)
    with torch.no_grad() if detach_teacher else torch.enable_grad():
        _, ht = forward_with_features(teacher, x)
    return F.mse_loss(hs, ht)


def loss_diversity_logdet(feat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    L_div: encourage feature covariance to have larger volume (avoid collapse).
    Use negative logdet of covariance -> minimize negative => maximize logdet.

    feat: [B, D]
    C = (F^T F)/B + eps I
    loss = -logdet(C)
    """
    B, D = feat.shape
    Fm = feat - feat.mean(dim=0, keepdim=True)
    C = (Fm.t() @ Fm) / max(B, 1)  # [D, D]
    C = C + eps * torch.eye(D, device=feat.device, dtype=feat.dtype)
    sign, logabsdet = torch.linalg.slogdet(C)
    # if sign<=0 something is wrong; clamp by eps should keep sign positive
    return -logabsdet


def loss_diversity_offdiag_cov(feat: torch.Tensor) -> torch.Tensor:
    """
    Alternative L_div: encourage decorrelation (Barlow/whitening-like):
      minimize ||offdiag(corr)||^2
    This discourages collapse and encourages diverse features.
    """
    B, D = feat.shape
    Fm = feat - feat.mean(dim=0, keepdim=True)
    C = (Fm.t() @ Fm) / max(B, 1)  # covariance
    diag = torch.diag(C)
    std = torch.sqrt(diag.clamp(min=1e-6))
    corr = C / (std[:, None] * std[None, :])
    off = corr - torch.diag(torch.diag(corr))
    return (off * off).mean()

@torch.no_grad()
def gn_safe_pseudo_replay(
    model: nn.Module,
    memory_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: torch.device,
    *,
    n_stat_batches: int = 2,
    noise_scale: float = 0.25,
    clamp: Optional[Tuple[float, float]] = None,
    make_labels: bool = False,
):
    """
    GN-safe pseudo replay (feature-conditioned, input-space sampling)

    Return:
      x_pseudo: [B, *input_shape]
      y_pseudo: optional (argmax probs)
    """
    model.eval()

    # --- infer input stats from real memory data ---
    xs = []
    it = iter(memory_loader)
    for _ in range(max(1, n_stat_batches)):
        try:
            xb, _ = next(it)
        except StopIteration:
            it = iter(memory_loader)
            xb, _ = next(it)
        xs.append(xb)

    X = torch.cat(xs, dim=0).to(device)  # [N, ...]
    # per-dimension stats (same shape as input excluding batch)
    mu = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp(min=1e-6)

    eps = torch.randn((batch_size, *X.shape[1:]), device=device)
    x_p = X[torch.randperm(128), :] + noise_scale * std * eps

    if clamp is not None:
        lo, hi = clamp
        x_p = x_p.clamp(lo, hi)

    if not make_labels:
        return x_p, None

    logits_p = forward_with_features(model, x_p)[0]
    y_p = torch.argmax(torch.softmax(logits_p, dim=1), dim=1)
    return x_p, y_p

# ============================================================
# 2) Sleep objective configuration
# ============================================================

@dataclass
class SleepObjectiveConfig:
    # Downregulation (B)
    down_alpha: float = 0.85

    # Sleep optimization steps
    sleep_steps: int = 100
    sleep_lr: float = 1e-3
    weight_decay: float = 0.0

    # Replay sizes
    mem_batch: int = 128
    pseudo_batch: int = 128    # 128

    lambda_importance: float = 0.5


    use_dormant_reactivation: bool = True
    dormant_threshold: float = 0.01       # < 1% of peak feature energy = dormant
    dormant_excitability_delta: float = 0.01   # bias increment for dormant neurons


    use_oscillation_structure: bool = True
    spindle_base_steps: int = 5       # base consolidation steps per task per spindle
    swr_steps: int = 10               # total SWR phase steps (intense boundary replay)
    swr_topk_frac: float = 0.3        # top-k uncertain samples selected for SWR
    rems_div_scale: float = 2.0       # L_div multiplier in REM-like phase
    rems_steps: int = 20              # steps in REM-like functional-forgetting phase


    adaptive_lr: bool = True
    lr_ratio_scale: float = 0.05      

    adaptive_merge: bool = True
    merge_quality_weight: float = 0.5  # how much quality signal affects merge_lam

    # Distillation
    lambda_distill: float = 0.5
    distill_T: float = 2.0
    lambda_feat: float = 0.0  # feature distill weight (optional)

    # Diversity (A)
    lambda_div: float = 0.05
    div_mode: str = "logdet"  # "logdet" or "offdiag"

    # Orthogonalization (C): projection is hard constraint (recommended)
    use_grad_projection: bool = True
    ortho_only_head: bool = True

    # Build/update task subspace basis
    basis_batches: int = 20
    basis_dim: int = 32

    use_pseudo: bool = True

    lambda_distill_min: float = 0.2
    lambda_distill_max: float = 0.8
    lambda_distill_mode: str = "linear"  # "linear" or "sqrt"
    num_tasks: int = 10  # 전체 task 수를 args에서 넘기면 가장 좋음
    lambda_old: float = 1.0

    # Fisher-weighted downregulation
    use_fisher_downreg: bool = True     # selective downreg based on Fisher importance
    fisher_protect_quantile: float = 0.7  # top 30% Fisher weights are protected
    alpha_protect: float = 0.95           # mild downreg for high-Fisher weights (two-tier)

    downreg_stabilize_steps: int = 10

    use_post_downreg_teacher: bool = True

    adaptive_spindle: bool = True
    spindle_base_steps_min: int = 2     # min steps/task when tasks are dissimilar
    spindle_base_steps_max: int = 10    # max steps/task when tasks are similar
    adaptive_sleep_steps: bool = True
    sleep_steps_min: int = 50           # steps when pressure is low
    sleep_steps_max: int = 200          # steps when pressure is high
    adaptive_lambda_imp: bool = True
    lambda_importance_min: float = 0.2
    lambda_importance_max: float = 2.0

# ============================================================
# Method A: Inter-task feature distance (interference proxy)
# ============================================================

@torch.no_grad()
def compute_inter_task_distance(
    per_task_mem_xs: list,
    model: nn.Module,
    device: torch.device,
    n_samples: int = 64,
) -> float:
    """
    Compute mean pairwise cosine distance between task feature centroids.

    Low distance → tasks share feature space → high interference → more spindle.
    High distance → tasks are separable → low interference → fewer spindle steps.

    Returns a value in [0, 1]:
      0 = all task centroids identical (maximum interference)
      1 = all task centroids orthogonal (zero interference)

    per_task_mem_xs: list of Tensor(N, C, T), channels-first (from _sleep_mem_xs)
    """
    if len(per_task_mem_xs) < 2:
        return 1.0  # only one task — no inter-task distance, use minimum spindle

    model.eval()
    centroids = []
    for x_task in per_task_mem_xs:
        n = min(n_samples, x_task.shape[0])
        x_s = x_task[:n].to(device)
        try:
            feat = model.feature(x_s) if hasattr(model, 'feature') else model(x_s)
        except Exception:
            return 1.0
        c = feat.mean(0)                          # (D,)
        if not torch.isfinite(c).all():           # skip NaN/Inf centroids
            continue
        norm = c.norm()
        centroids.append(c / norm if norm >= 1e-8 else c)

    if len(centroids) < 2:
        return 1.0

    # Mean pairwise cosine distance (= 1 - cosine_sim), normalised to [0,1]
    # cosine_sim ∈ [-1,1] → distance ∈ [0,1] after (1 - sim) / 2
    total, count = 0.0, 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            cos_sim = float((centroids[i] * centroids[j]).sum().clamp(-1.0, 1.0))
            total += (1.0 - cos_sim) / 2.0        # map to [0, 1]
            count += 1

    result = total / count if count > 0 else 1.0
    return result if (result == result) else 1.0   # guard residual NaN



def loss_importance_regularize(
    model_sleep: nn.Module,
    theta_wake_sd: Dict[str, torch.Tensor],
    importance_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Penalize sleep from changing parameters that are important for past tasks.

    L_imp = Σ_i  importance_i * (θ_sleep_i − θ_wake_i)²

    Corresponds to synaptic tagging & capture: only un-tagged (low-importance)
    synapses undergo full sleep plasticity; tagged ones are protected.

    importance_dict can come from:
      EWC  → self.importances (Fisher, per-task max)
      MAS  → self.importance  (output norm sensitivity)
      SI   → self.ewc_data[1] (path-integral, reshaped)
      DER  → compute_fisher_soft_target (soft-logit Fisher)
      else → compute_fisher_diagonal on sleep memory (fallback)
    """
    L = torch.zeros((), device=device)
    sd = {k: v.to(device) for k, v in model_sleep.state_dict().items()}
    for name, imp in importance_dict.items():
        if name not in sd or name not in theta_wake_sd:
            continue
        imp_d = imp.to(device)
        diff = sd[name] - theta_wake_sd[name].to(device)
        # Guard: imp and diff must have the same shape
        if imp_d.shape != diff.shape:
            min_shape = tuple(min(a, b) for a, b in zip(imp_d.shape, diff.shape))
            slices = tuple(slice(0, s) for s in min_shape)
            imp_d = imp_d[slices]
            diff  = diff[slices]
        L = L + (imp_d * diff.pow(2)).sum()
    return L



@torch.no_grad()
def reactivate_dormant_neurons(
    model: nn.Module,
    memory_loader,
    device: torch.device,
    dormant_threshold: float = 0.01,
    excitability_delta: float = 0.01,
    n_batches: int = 3,
) -> int:
    """
    Homeostatic plasticity (Turrigiano & Nelson, 2004):
    Neurons whose mean |activation| falls below dormant_threshold × peak
    receive an intrinsic excitability increase: bias += excitability_delta.

    Modifies biases of Conv1d / Linear layers whose output dim matches dormant
    feature dimensions. This is more biologically accurate than adding noise to
    head weights — it raises the neuron's baseline firing threshold, not the
    downstream synaptic weight.

    Returns the number of dormant dimensions reactivated.
    """
    model.eval()

    # 1) Collect feature activations over a few batches
    all_feats = []
    it = iter(memory_loader)
    for _ in range(n_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            break
        with torch.no_grad():
            if hasattr(model, 'feature'):
                f = model.feature(xb.to(device))
            else:
                f = model(xb.to(device))
        all_feats.append(f.detach().cpu())

    if not all_feats:
        return 0

    feats = torch.cat(all_feats, dim=0)          # (N, D)
    energy = torch.abs(feats).mean(dim=0)         # (D,)  mean |activation| per dim
    threshold = energy.max().item() * dormant_threshold
    dormant_mask = (energy <= threshold)          # (D,) bool

    n_dormant = int(dormant_mask.sum().item())
    if n_dormant == 0:
        return 0

    D = feats.shape[1]

    reactivated = False
    for name, p in model.named_parameters():
        if 'head' in name:
            continue                              # skip classifier head
        if 'bias' not in name:
            continue
        if p.dim() != 1 or p.shape[0] != D:
            continue
        # Found the bias vector matching the feature dimension
        mask_dev = dormant_mask.to(p.device)
        p.data[mask_dev] += excitability_delta
        reactivated = True
        break  # one pass: first matching encoder bias layer

    if not reactivated:
        # Fallback: if no bias matches D exactly, add a small positive
        # excitation to ALL encoder biases (weaker, still homeostatic)
        for name, p in model.named_parameters():
            if 'head' in name or 'bias' not in name or p.dim() != 1:
                continue
            p.data += excitability_delta * 0.1

    return n_dormant


# ============================================================
# B Phase 3: SWR — collect high-uncertainty samples
# ============================================================

def collect_swr_loader(
    model: nn.Module,
    memory_loader,
    device: torch.device,
    topk_frac: float = 0.3,
    batch_size: int = 64,
) -> Optional[object]:
    """
    Sharp-Wave Ripple (SWR) phase: select samples near decision boundaries
    (highest prediction entropy) for intense, brief replay.

    In neuroscience, SWRs replay the most salient / uncertain memories —
    those that are 'at risk' of being forgotten.  High-entropy predictions
    identify samples where the model is least confident = most at risk.

    Returns a DataLoader of the top-k uncertain samples, or None if too few.
    """
    from torch.utils.data import TensorDataset, DataLoader

    model.eval()
    all_x, all_y, all_ent = [], [], []
    with torch.no_grad():
        for xb, yb in memory_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs  = torch.softmax(logits, dim=1)
            # Shannon entropy: H = -Σ p log p
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)  # (B,)
            all_x.append(xb.cpu())
            all_y.append(yb.cpu())
            all_ent.append(entropy.cpu())

    if not all_x:
        return None

    X   = torch.cat(all_x,   dim=0)
    Y   = torch.cat(all_y,   dim=0)
    Ent = torch.cat(all_ent, dim=0)

    k = max(1, int(len(X) * topk_frac))
    _, top_idx = torch.topk(Ent, k)

    X_swr = X[top_idx]
    Y_swr = Y[top_idx]

    if len(X_swr) < 2:
        return None

    bs = min(batch_size, len(X_swr))
    return DataLoader(TensorDataset(X_swr, Y_swr), batch_size=bs, shuffle=True)


# ============================================================
# C: Compute adaptive merge_lam (functional forgetting)
# ============================================================

@torch.no_grad()
def compute_adaptive_merge_lam(
    model_sleep: nn.Module,
    teacher_wake: nn.Module,
    memory_loader,
    device: torch.device,
    base_lam: float = 0.5,
    quality_weight: float = 0.5,
) -> float:
    """
    Adaptive merge_lam based on sleep consolidation quality.

    Functional Forgetting perspective (Crick & Mitchison; schema theory):
    - Sleep should be allowed to CHANGE the model when it has successfully
      abstracted / generalized (accuracy preserved + diversity increased).
    - Sleep should be CONSERVATIVE when it has degraded task performance
      (it may be over-writing task-specific knowledge).

    merge_lam = base_lam
                + quality_weight * accuracy_gain_fraction   (sleep improved → higher lam)
                + quality_weight * diversity_gain_fraction  (schema abstraction → higher lam)
    Clamped to [0.1, 0.9].

    accuracy_gain_fraction ∈ [-1, +1]:
        +1 = sleep perfectly preserved all tasks
        -1 = sleep degraded all tasks
    diversity_gain_fraction ∈ [0, 1]:
        +1 = large diversity increase (schema formation)
         0 = no change
    """
    model_sleep.eval()
    teacher_wake.eval()

    correct_sleep, correct_wake, total = 0, 0, 0
    feat_sleep_list, feat_wake_list = [], []

    for xb, yb in memory_loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            out_s = model_sleep(xb)
            out_w = teacher_wake(xb)
        correct_sleep += out_s.argmax(1).eq(yb).sum().item()
        correct_wake  += out_w.argmax(1).eq(yb).sum().item()
        total += len(yb)

        # Collect features for diversity comparison
        if hasattr(model_sleep, 'feature'):
            feat_sleep_list.append(model_sleep.feature(xb).detach())
            feat_wake_list.append(teacher_wake.feature(xb).detach())

    if total == 0:
        return base_lam

    acc_sleep = correct_sleep / total
    acc_wake  = correct_wake  / total
    # Normalised gain: +1 if sleep matched wake, -1 if sleep scored 0
    acc_gain = (acc_sleep - acc_wake) / (acc_wake + 1e-6)
    acc_gain = float(max(-1.0, min(1.0, acc_gain)))

    # Diversity gain: eff_rank increase = schema formation signal
    div_gain = 0.0
    if feat_sleep_list and feat_wake_list:
        def _eff_rank(feats):
            F = torch.cat(feats, dim=0)
            F = F.reshape(F.shape[0], -1)          # flatten any trailing dims
            valid = torch.isfinite(F).all(dim=-1)  # drop NaN/Inf rows
            F = F[valid]
            if F.shape[0] < 2:
                return 1.0
            Fm = F - F.mean(0, keepdim=True)
            C  = (Fm.T @ Fm) / max(Fm.shape[0], 1)
            C  = C + 1e-6 * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
            try:
                ev = torch.linalg.eigvalsh(C).clamp(min=1e-6)
            except Exception:
                return 1.0
            p  = ev / ev.sum()
            return float(torch.exp(-(p * p.log()).sum()).item())

        er_sleep = _eff_rank(feat_sleep_list)
        er_wake  = _eff_rank(feat_wake_list)
        div_gain = float(max(0.0, (er_sleep - er_wake) / (er_wake + 1e-6)))
        div_gain = min(1.0, div_gain)

    lam = base_lam + quality_weight * max(0.0, acc_gain) + quality_weight * div_gain
    lam = float(max(base_lam, min(0.9, lam)))
    return lam


def _flatten_grads(model: torch.nn.Module, only_head: bool = False) -> torch.Tensor:
    """
    Collect all gradients into a single 1-D tensor.
    only_head=True: head (classifier) 파라미터만 사용
    """
    grads = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if only_head and "head" not in name:
            continue
        grads.append(p.grad.detach().reshape(-1))
    if len(grads) == 0:
        return torch.zeros(1, device=next(model.parameters()).device)
    return torch.cat(grads, dim=0)


def _flat_grad_norm(model: nn.Module, only_head: bool = False) -> float:
    g = _flatten_grads(model, only_head=only_head)
    return float(g.norm().detach().cpu().item())

# ============================================================
# 3) θ_final computation: explicit objective optimization
# ============================================================

def compute_theta_final_sleep_objective(
    model_wake: nn.Module,
    memory_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    bank: TaskSubspaceBank,
    task_id: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: torch.device,
    cfg: SleepObjectiveConfig,
    theta_prev: Optional[Dict[str, torch.Tensor]] = None,
    per_task_loaders: Optional[list] = None,
    importance_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns θ_final (state_dict) after minimising the sleep objective.
    model_wake is NOT modified — sleep runs on an internal copy (model_sleep).

    When cfg.use_oscillation_structure=True, the sleep loop is structured as a
    4-phase NREM→REM oscillation hierarchy (B):

      Phase 1 — Slow Oscillation / Down-state
        • Downregulation (SHY: global synaptic scaling down)
        • Dormant neuron reactivation (homeostatic plasticity, 방안 5)

      Phase 2 — Sleep Spindles
        • Per-task focused rehearsal with AGE-WEIGHTED steps (방안 4 ⊂ B)
          Older tasks get proportionally more spindle steps — counters
          recency bias that would otherwise leave early tasks unprotected.
        • Adaptive lr: spindle intensity ∝ g_plastic / g_retain (방안 2 ⊂ B)
          High plasticity load → slower, more careful spindle consolidation.
        • Importance regularisation (방안 1): L_imp penalises moving
          task-critical parameters (EWC/MAS/SI/DER importance).

      Phase 3 — Sharp-Wave Ripples (SWR)
        • Brief, intense replay of high-uncertainty (boundary) samples.
        • These are the memories most at risk — SWRs prioritise them.

      Phase 4 — REM-like functional forgetting (C)
        • Diversity loss dominates (λ_div × rems_div_scale).
        • Higher pseudo-replay noise → schema abstraction.
        • Allows episodic details to fade while preserving statistical
          structure (Crick & Mitchison reverse-learning; schema theory).

    importance_dict: {param_name: importance_tensor} from base agent
                     (EWC Fisher / MAS Ω / SI path-integral / DER soft-Fisher)
    """
    model_wake.to(device)

    T_distill = float(getattr(cfg, "distill_T", 2.0))
    lambda_old = float(getattr(cfg, "lambda_old", 1.0))
    lambda_imp = float(getattr(cfg, "lambda_importance", 0.5))
    use_osc    = bool(getattr(cfg, "use_oscillation_structure", True))

    _norm_pressure = getattr(cfg, '_norm_pressure', 0.0)
    if getattr(cfg, 'adaptive_lambda_imp', True) and _norm_pressure > 0:
        li_min = getattr(cfg, 'lambda_importance_min', 0.2)
        li_max = getattr(cfg, 'lambda_importance_max', 2.0)
        lambda_imp = li_min + (li_max - li_min) * _norm_pressure
        print(f"  [Method B / Imp] norm_pressure={_norm_pressure:.3f} "
              f"→ lambda_imp={lambda_imp:.3f} (range [{li_min},{li_max}])")

    def _schedule_lambda_distill(task_id: int) -> float:
        lam_min = getattr(cfg, "lambda_distill_min", cfg.lambda_distill)
        lam_max = getattr(cfg, "lambda_distill_max", cfg.lambda_distill)
        T = max(1, int(getattr(cfg, "num_tasks", task_id + 1)))
        t = min(1.0, float(task_id) / float(max(1, T - 1)))
        if getattr(cfg, "lambda_distill_mode", "linear") == "sqrt":
            t = t ** 0.5
        return lam_min + (lam_max - lam_min) * t

    def _one_step(xb, yb, opt, lambda_div_scale=1.0,
                  distill_scale=1.0, old_scale=1.0, log_prefix=None):
        """Single optimisation step shared across all phases.

        distill_scale: multiplier on lam_dist*L_dist  (0.0 in Phase 1.5 to
                       prevent L_dist from reversing Phase 1 downregulation)
        old_scale:     multiplier on lambda_old*L_old (0.0 in Phase 1.5)
        """
        xb, yb = xb.to(device), yb.to(device)

        if cfg.use_pseudo and cfg.pseudo_batch > 0 and _pseudo_mu is not None:
            _eps = torch.randn((cfg.pseudo_batch, *xb.shape[1:]), device=device)
            xp = (_pseudo_mu
                  + getattr(cfg, "pseudo_noise_scale", 0.25) * _pseudo_std * _eps
                  ).detach()
            x_all = torch.cat([xb, xp], dim=0)
        else:
            x_all = xb

        with torch.no_grad():
            t_wake_logits = forward_with_features(teacher_wake, x_all)[0]

        s_logits_all, feat_all = forward_with_features(model_sleep, x_all)
        s_logits_xb = s_logits_all[:xb.shape[0]]

        L_cons = F.cross_entropy(s_logits_xb, yb)

        L_dist = F.kl_div(
            F.log_softmax(s_logits_all / max(T_distill, 1e-6), dim=1),
            F.softmax(t_wake_logits / max(T_distill, 1e-6), dim=1),
            reduction="batchmean",
        ) * (T_distill ** 2)

        L_old = torch.zeros((), device=device)
        if teacher_prev is not None:
            teacher_prev.to(device)
            with torch.no_grad():
                f_prev = (teacher_prev.feature(xb)
                          if hasattr(teacher_prev, 'feature')
                          else forward_with_features(teacher_prev, xb)[1])
            teacher_prev.cpu()
            f_sleep = (model_sleep.feature(xb)
                       if hasattr(model_sleep, 'feature')
                       else feat_all[:xb.shape[0]])
            L_old = F.mse_loss(f_sleep, f_prev.detach())

        L_div = (loss_diversity_offdiag_cov(feat_all)
                 if cfg.div_mode == "offdiag"
                 else loss_diversity_logdet(feat_all))

        L_feat = torch.zeros((), device=device)
        if cfg.lambda_feat > 0:
            with torch.no_grad():
                _, ht = forward_with_features(teacher_wake, x_all)
            L_feat = F.mse_loss(feat_all, ht.detach())

        # 방안 1: importance regularisation
        L_imp = torch.zeros((), device=device)
        if importance_dict is not None and lambda_imp > 0:
            L_imp = loss_importance_regularize(
                model_sleep, theta_wake_sd, importance_dict, device)

        lam_dist = _schedule_lambda_distill(task_id)
        L = (L_cons
             + lam_dist * distill_scale * L_dist
             + lambda_old * old_scale * L_old
             + cfg.lambda_feat * L_feat
             + cfg.lambda_div * lambda_div_scale * L_div
             + lambda_imp * L_imp)

        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(L):
            return
        L.backward()

        if cfg.use_grad_projection:
            orthogonalize_gradient_in_place(
                model_sleep, bank, upto_task_id=task_id - 1,
                only_head=cfg.ortho_only_head)
        opt.step()

        if log_prefix is not None:
            print(f"  [{log_prefix}] L_cons={L_cons.item():.4f} "
                  f"L_dist={L_dist.item():.4f} L_old={L_old.item():.4f} "
                  f"L_div={L_div.item():.4f} L_imp={L_imp.item():.4f}")

    teacher_wake = copy.deepcopy(model_wake).to(device)
    teacher_wake.eval()
    for p in teacher_wake.parameters():
        p.requires_grad_(False)

    teacher_prev = None
    if theta_prev is not None and task_id > 0:
        teacher_prev = copy.deepcopy(model_wake).cpu()
        cur_sd = teacher_prev.state_dict()
        matching_sd = {k: v.cpu() for k, v in theta_prev.items()
                       if k in cur_sd and cur_sd[k].shape == v.shape}
        teacher_prev.load_state_dict(matching_sd, strict=False)
        teacher_prev.eval()
        for p in teacher_prev.parameters():
            p.requires_grad_(False)

    model_sleep = copy.deepcopy(model_wake).to(device)
    model_sleep.train()

    # Cache θ_wake state_dict for importance penalty reference
    theta_wake_sd = {k: v.detach().clone()
                     for k, v in teacher_wake.state_dict().items()}

    print(f"  [Phase 1 / Down-state] task={task_id} — downregulation + reactivation")
    if task_id > 0 and getattr(cfg, 'use_fisher_downreg', True):
        fisher_for_downreg = compute_fisher_diagonal(
            model_wake, memory_loader, n_batches=5, device=device)
        downregulation_fisher_weighted(
            model_sleep, alpha=cfg.down_alpha, fisher=fisher_for_downreg,
            protect_quantile=getattr(cfg, 'fisher_protect_quantile', 0.7),
            alpha_protect=getattr(cfg, 'alpha_protect', 0.95),
        )
    else:
        downregulation(model_sleep, alpha=cfg.down_alpha)

    if getattr(cfg, 'use_dormant_reactivation', True):
        n_reactivated = reactivate_dormant_neurons(
            model=model_sleep,
            memory_loader=memory_loader,
            device=device,
            dormant_threshold=getattr(cfg, 'dormant_threshold', 0.01),
            excitability_delta=getattr(cfg, 'dormant_excitability_delta', 0.01),
        )
        print(f"  [Phase 1 / Reactivation] {n_reactivated} dormant dims reactivated")

    build_task_gradient_subspace(
        model=model_sleep,
        data_iter=memory_loader,
        task_id=task_id,
        bank=bank,
        num_batches=cfg.basis_batches,
        basis_dim=cfg.basis_dim,
        only_head=cfg.ortho_only_head,
    )


    _pseudo_mu: Optional[torch.Tensor] = None
    _pseudo_std: Optional[torch.Tensor] = None
    if cfg.use_pseudo and cfg.pseudo_batch > 0:
        _xs, _it = [], iter(memory_loader)
        for _ in range(max(1, getattr(cfg, "pseudo_stat_batches", 20))):
            try:
                _xb, _ = next(_it)
            except StopIteration:
                _it = iter(memory_loader)
                _xb, _ = next(_it)
            _xs.append(_xb)
        _X = torch.cat(_xs, dim=0).to(device)
        _pseudo_mu  = _X.mean(dim=0, keepdim=True)
        _pseudo_std = _X.std(dim=0, keepdim=True).clamp(min=1e-6)
        del _X, _xs, _it
        torch.cuda.empty_cache()


    effective_lr = cfg.sleep_lr
    if getattr(cfg, 'adaptive_lr', True) and hasattr(cfg, '_g_plastic_prev'):
        import math as _math
        g_p = getattr(cfg, '_g_plastic_prev', 1.0)
        g_r = getattr(cfg, '_g_retain_prev',  1.0)
        # Log-normalized: raw ratio g_p/g_r can be O(10^4-10^5) making
        # 1/(1+scale*ratio) ≈ 0, killing all sleep optimization.
        # Use log-space excess instead: range ~0-15 nats, well-behaved.
        log_excess = max(0.0, _math.log1p(abs(g_p)) - _math.log1p(abs(g_r)))
        norm = min(1.0, log_excess / 10.0)   # 10 nats → full plastic range
        effective_lr = cfg.sleep_lr * (1.0 - 0.5 * norm)   # at most 50% reduction
        effective_lr = max(effective_lr, cfg.sleep_lr * 0.3)
        print(f"  [Adaptive lr] g_plastic={g_p:.3f} g_retain={g_r:.3f} "
              f"log_excess={log_excess:.2f} norm={norm:.2f} "
              f"→ lr={effective_lr:.2e} (base={cfg.sleep_lr:.2e})")

    opt = torch.optim.SGD(
        model_sleep.parameters(),
        lr=effective_lr,
        momentum=0.0,
        weight_decay=cfg.weight_decay,
    )


    _stabilize_steps = getattr(cfg, 'downreg_stabilize_steps', 10)
    if _stabilize_steps > 0 and task_id > 0:
        stab_iter = iter(memory_loader)
        print(f"  [Phase 1.5 / Stabilize] {_stabilize_steps} steps "
              f"(L_cons only, distill=0, old=0)")
        for _s_idx in range(_stabilize_steps):
            try:
                xb_s, yb_s = next(stab_iter)
            except StopIteration:
                stab_iter = iter(memory_loader)
                xb_s, yb_s = next(stab_iter)
            _one_step(xb_s, yb_s, opt,
                      distill_scale=0.0, old_scale=0.0,
                      log_prefix=f"stabilize {_s_idx}" if _s_idx == 0 else None)


    if getattr(cfg, 'use_post_downreg_teacher', True) and task_id > 0:
        teacher_wake.cpu()
        del teacher_wake
        torch.cuda.empty_cache()
        teacher_wake = copy.deepcopy(model_sleep).to(device)
        teacher_wake.eval()
        for p in teacher_wake.parameters():
            p.requires_grad_(False)
        print("  [Method 2] teacher_wake → post-downreg snapshot")


    _inter_task_dist = getattr(cfg, '_inter_task_dist', 1.0)
    if not isinstance(_inter_task_dist, (int, float)) or not (0.0 <= _inter_task_dist <= 1.0):
        _inter_task_dist = 1.0
    if getattr(cfg, 'adaptive_spindle', True) and task_id > 0:
        s_min = getattr(cfg, 'spindle_base_steps_min', 2)
        s_max = getattr(cfg, 'spindle_base_steps_max', 10)
        # dist=0 → max steps; dist=1 → min steps (linear interpolation)
        _dyn_spindle_base = int(round(s_max - (s_max - s_min) * _inter_task_dist))
        _dyn_spindle_base = max(s_min, min(s_max, _dyn_spindle_base))
        print(f"  [Method A / Spindle] inter_task_dist={_inter_task_dist:.3f} "
              f"→ spindle_base={_dyn_spindle_base} (range [{s_min},{s_max}])")
    else:
        _dyn_spindle_base = getattr(cfg, 'spindle_base_steps', 5)


    if getattr(cfg, 'adaptive_sleep_steps', True) and _norm_pressure > 0:
        st_min = getattr(cfg, 'sleep_steps_min', 50)
        st_max = getattr(cfg, 'sleep_steps_max', 200)
        _dyn_sleep_steps = int(round(st_min + (st_max - st_min) * _norm_pressure))
        _dyn_sleep_steps = max(st_min, min(st_max, _dyn_sleep_steps))
        print(f"  [Method B / Steps] norm_pressure={_norm_pressure:.3f} "
              f"→ sleep_steps={_dyn_sleep_steps} (range [{st_min},{st_max}])")
    else:
        _dyn_sleep_steps = cfg.sleep_steps

    if not use_osc:
        # ── Legacy flat loop (oscillation disabled) ──────────────────
        mem_iter = iter(memory_loader)
        for _step in range(_dyn_sleep_steps):
            try:
                xb, yb = next(mem_iter)
            except StopIteration:
                mem_iter = iter(memory_loader)
                xb, yb = next(mem_iter)
            log = f"sleep step {_step}" if _step < 3 else None
            _one_step(xb, yb, opt, log_prefix=log)
    else:

        mem_iter = iter(memory_loader)
        print(f"  [Phase 2a / Global] {_dyn_sleep_steps} steps (all tasks, combined)")
        for _step in range(_dyn_sleep_steps):
            try:
                xb, yb = next(mem_iter)
            except StopIteration:
                mem_iter = iter(memory_loader)
                xb, yb = next(mem_iter)
            _one_step(xb, yb, opt,
                      log_prefix=f"global step {_step}" if _step < 2 else None)


        spindle_base = _dyn_spindle_base   # Method A dynamic value
        if per_task_loaders is not None and task_id > 0:
            n_loaders = len(per_task_loaders)
            total_spindle_steps = sum(
                spindle_base * max(1, task_id - old_tid)
                for old_tid in range(min(task_id, n_loaders))
            )
            print(f"  [Phase 2b / Spindle] {min(task_id, n_loaders)} old tasks, "
                  f"age-weighted (base={spindle_base}) — "
                  f"total≈{total_spindle_steps}")
            for old_tid in range(task_id):   # current task excluded: Phase 2a already covered it
                if old_tid >= n_loaders:
                    break
                # Age weight: oldest task (old_tid=0) gets most extra steps.
                age_w = max(1, task_id - old_tid)
                n_spindle_steps = spindle_base * age_w
                old_iter = iter(per_task_loaders[old_tid])
                for s_idx in range(n_spindle_steps):
                    try:
                        xb_f, yb_f = next(old_iter)
                    except StopIteration:
                        old_iter = iter(per_task_loaders[old_tid])
                        xb_f, yb_f = next(old_iter)
                    log = (f"spindle t{old_tid} s{s_idx}" if s_idx == 0 else None)
                    _one_step(xb_f, yb_f, opt, log_prefix=log)

                    # Gradient orthogonality: project away from older task subspaces
                    if cfg.use_grad_projection and old_tid > 0:
                        orthogonalize_gradient_in_place(
                            model_sleep, bank,
                            upto_task_id=old_tid - 1,
                            only_head=cfg.ortho_only_head)

        swr_steps = getattr(cfg, 'swr_steps', 10)
        if swr_steps > 0:
            swr_loader = collect_swr_loader(
                model=model_sleep,
                memory_loader=memory_loader,
                device=device,
                topk_frac=getattr(cfg, 'swr_topk_frac', 0.3),
                batch_size=min(64, cfg.mem_batch),
            )
            if swr_loader is not None:
                swr_iter = iter(swr_loader)
                # SWR uses a higher lr momentarily (brief intense burst)
                for pg in opt.param_groups:
                    pg['lr'] = effective_lr * 2.0
                print(f"  [Phase 3 / SWR] {swr_steps} steps on "
                      f"top-{int(cfg.swr_topk_frac*100)}% uncertain samples "
                      f"(lr×2={effective_lr*2:.2e})")
                for s_idx in range(swr_steps):
                    try:
                        xb_s, yb_s = next(swr_iter)
                    except StopIteration:
                        swr_iter = iter(swr_loader)
                        xb_s, yb_s = next(swr_iter)
                    _one_step(xb_s, yb_s, opt,
                              log_prefix=f"SWR {s_idx}" if s_idx == 0 else None)
                # Restore normal lr
                for pg in opt.param_groups:
                    pg['lr'] = effective_lr
            else:
                print("  [Phase 3 / SWR] skipped (not enough samples)")

        rems_steps = getattr(cfg, 'rems_steps', 20)
        rems_div_scale = getattr(cfg, 'rems_div_scale', 2.0)
        if rems_steps > 0 and cfg.use_pseudo and _pseudo_mu is not None:
            # Boost noise for pseudo-replay in REM phase (more generalization)
            _orig_noise = getattr(cfg, "pseudo_noise_scale", 0.25)
            cfg.__dict__["pseudo_noise_scale"] = _orig_noise * 2.0
            mem_iter_rem = iter(memory_loader)
            print(f"  [Phase 4 / REM] {rems_steps} steps, "
                  f"L_div×{rems_div_scale}, noise×2 (schema abstraction)")
            for r_idx in range(rems_steps):
                try:
                    xb_r, yb_r = next(mem_iter_rem)
                except StopIteration:
                    mem_iter_rem = iter(memory_loader)
                    xb_r, yb_r = next(mem_iter_rem)
                _one_step(xb_r, yb_r, opt,
                          lambda_div_scale=rems_div_scale,
                          log_prefix=f"REM {r_idx}" if r_idx == 0 else None)
            cfg.__dict__["pseudo_noise_scale"] = _orig_noise  # restore


    theta_final = copy_state_dict(model_sleep)
    return theta_final


@torch.no_grad()
def merge_theta_wake_and_final(model: nn.Module, theta_final: Dict[str, torch.Tensor], lam_merge: float = 0.3) -> None:
    """
    θ <- (1-λ) θ_wake + λ θ_final
    """
    cur = model.state_dict()
    merged = {}
    for k, v in cur.items():
        fw = theta_final[k].to(v.device)
        merged[k] = (1.0 - lam_merge) * v + lam_merge * fw
    model.load_state_dict(merged, strict=True)

