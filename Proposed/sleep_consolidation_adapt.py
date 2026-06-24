"""
Post-task sleep consolidation for any CL agent.

Usage (via factory — preferred):
    agent = make_sleep_cl_agent(model, args)   # uses args.sleep_base_agent

Usage (direct subclass):
    class SleepER(SleepCLMixin, ExperienceReplay): pass
    class SleepLwF(SleepCLMixin, LwF): pass

Design:
- Intercepts learn_task(): calls super().learn_task() (wake) then runs sleep.
- Maintains a small internal episodic memory (channels-first, N×C×T) filled
  from each task's training data.  Used when the base agent has no buffer.
- If the base agent exposes a replay buffer (ER/ASER/DER/Herding/…), that
  buffer is used as primary sleep replay source; internal memory is fallback.
- After sleep consolidation the consolidated weights are saved as the
  checkpoint so evaluate() and the next wake-phase start from them.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from agents._sleep_func_v3 import (
    SleepObjectiveConfig,
    compute_theta_final_sleep_objective,
    merge_theta_wake_and_final,
    compute_adaptive_merge_lam,
)
from agents._sleep_opt import TaskSubspaceBank, compute_fisher_diagonal
from agents.norm_replace import replace_bn_with_gn


class SleepCLMixin:
    """
    Mixin that adds sleep-phase consolidation on top of any BaseLearner subclass.

    MRO must place SleepCLMixin BEFORE the base agent:
        class SleepCL_ER(SleepCLMixin, ExperienceReplay): pass
    """

    def __init__(self, model, args):
        # Initialize the base CL agent first (sets self.model, self.optimizer,
        # self.device, self.args, self.buffer, …)
        super().__init__(model, args)
        self._init_sleep_components(args)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_sleep_components(self, args):
        """Set up sleep-specific state after the base agent is initialised."""

        # Optional: replace BN → GN for training-batch-size robustness.
        # This must happen BEFORE any sleep training; we also recreate the
        # optimizer so it references the updated parameters.
        if getattr(args, "replace_bn_with_gn", True):
            self.model = replace_bn_with_gn(self.model, num_groups=32)
            self.model.to(self.device)
            # Recreate optimizer: BN params were swapped for GN params.
            from utils.optimizer import set_optimizer
            self.optimizer = set_optimizer(self.model, args)

        # Task-subspace bank (for gradient orthogonalisation in sleep).
        self.bank = TaskSubspaceBank()

        # Sleep objective hyperparameters.
        self.sleep_obj_cfg = SleepObjectiveConfig(
            down_alpha=getattr(args, "down_alpha", 0.85),
            sleep_steps=getattr(args, "sleep_steps", 100),
            sleep_lr=getattr(args, "sleep_lr", 1e-3),
            lambda_distill=getattr(args, "lambda_distill", 0.5),
            lambda_div=getattr(args, "lambda_div", 0.2),
            distill_T=2.0,
            lambda_old=getattr(args, "lambda_old", 1.0),
            use_fisher_downreg=True,
            fisher_protect_quantile=getattr(args, "fisher_protect_quantile", 0.7),
            alpha_protect=getattr(args, "alpha_protect", 0.95),
            focused_steps_per_task=getattr(args, "focused_steps_per_task", 10),
        )
        self.sleep_merge_lam = getattr(args, "merge_lam", 0.5)

        # Internal sleep memory: one Tensor per task, already in (N, C, T).
        self._sleep_mem_xs: list = []   # list[Tensor(n, C, T)]
        self._sleep_mem_ys: list = []   # list[Tensor(n,)]
        self._sleep_samples_per_class: int = getattr(args, "sleep_samples_per_class", 20)

        # Per-task plasticity log: accumulated across tasks in one run.
        # Each entry: {'task_id', 'pre', 'post', 'delta', 'pct_delta'}
        # Collected by exp.py after the run to be saved in result.pkl.
        self._sleep_plasticity_log: list = []

    # ------------------------------------------------------------------
    # Core override: learn_task = wake (base agent) + sleep (mixin)
    # ------------------------------------------------------------------

    def learn_task(self, task):
        """Wake phase via super() then sleep consolidation."""
        (x_train, y_train), _, _ = task

        # 1) Save pre-wake model as old knowledge anchor.
        #    This snapshot is passed to sleep consolidation so distillation
        #    is anchored to the model BEFORE wake training (which may partially
        #    overwrite old-task knowledge). theta_prev is NOT modified below.
        self._theta_prev = {k: v.detach().clone()
                            for k, v in self.model.state_dict().items()}

        # 2) Wake: delegate to the base CL agent
        super().learn_task(task)
        task_id = self.task_now   # updated by before_task() inside super()

        # 2) Collect a balanced subset from current-task data into sleep memory
        self._collect_sleep_memory(x_train, y_train)

        # 3) Build DataLoader for sleep (buffer preferred, else internal memory)
        replay_loader = self._build_sleep_replay_loader()
        if replay_loader is None:
            print(f"[SleepCL] task={task_id} | No replay data available — skipping sleep.")
            return

        # 4) Plasticity snapshot BEFORE sleep
        pre = self._plasticity_snapshot(x_train, y_train)

        # 5) Sleep consolidation
        self._do_sleep_consolidation(replay_loader, task_id)

        # 6) Plasticity snapshot AFTER sleep  → compare with pre
        post = self._plasticity_snapshot(x_train, y_train)
        self._log_sleep_plasticity_delta(pre, post, task_id)

        # 7) Overwrite checkpoint with consolidated weights so that:
        #    - evaluate() reloads the consolidated model
        #    - next task's wake starts from the consolidated model
        torch.save(self.model.state_dict(), self.ckpt_path)
        print(f"[SleepCL] task={task_id} | Checkpoint updated with consolidated weights.")

    # ------------------------------------------------------------------
    # Internal sleep memory
    # ------------------------------------------------------------------

    def _collect_sleep_memory(self, x_train, y_train):
        """
        Sample up to `_sleep_samples_per_class` examples per class from the
        current task's training data.

        Stores tensors in (N, C, T) channels-first format, consistent with
        what SingleHeadModel.forward now expects:
          - CNNEncoder.forward no longer transposes internally (transpose removed)
          - Wake training does permute(0,2,1) before model call
          - Sleep functions call model(x) directly, so x must be (N, C, T)
        """
        n_per = self._sleep_samples_per_class
        y_np = (y_train.cpu().numpy() if isinstance(y_train, torch.Tensor)
                else np.asarray(y_train))
        x_np = (x_train.cpu().numpy() if isinstance(x_train, torch.Tensor)
                else np.asarray(x_train))
        # Chapman Time-IL pickle may produce object_ arrays when ECG segments
        # have varying lengths or are stored as Python lists.  Convert to float32
        # so torch.FloatTensor() can accept the array.
        if x_np.dtype == object:
            x_np = np.array(list(x_np), dtype=np.float32)

        xs, ys = [], []
        for cls in np.unique(y_np):
            idxs = np.where(y_np == cls)[0]
            chosen = np.random.choice(idxs, min(n_per, len(idxs)), replace=False)
            # x_np is (N, T, C) — permute to (N, C, T) for direct model call
            x_t = torch.FloatTensor(x_np[chosen]).permute(0, 2, 1)
            xs.append(x_t)
            ys.append(torch.full((len(chosen),), int(cls), dtype=torch.long))

        if xs:
            self._sleep_mem_xs.append(torch.cat(xs, dim=0))
            self._sleep_mem_ys.append(torch.cat(ys, dim=0))

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def _build_sleep_replay_loader(self):
        """
        Build a DataLoader for sleep consolidation.

        Priority order (REVERSED from buffer-first):
          1. Internal sleep memory  — always preferred because it guarantees
             equal samples per class per task (N per class, all tasks equally
             represented). The ER buffer may be recency-biased.
          2. Base agent's replay buffer — fallback if internal memory is empty
             (e.g. first task where no prior memory exists yet).

        All batches are in (N, C, T) — channels-first.
        """
        if self._sleep_mem_xs:
            X = torch.cat(self._sleep_mem_xs, dim=0)  # (N_total, C, T)
            Y = torch.cat(self._sleep_mem_ys, dim=0)  # (N_total,)
            return DataLoader(
                TensorDataset(X, Y),
                batch_size=self.args.batch_size,
                shuffle=True,
            )

        # Fallback: base agent's replay buffer
        buf = getattr(self, "buffer", None)
        if buf is not None:
            try:
                loader = self._loader_from_buffer(buf)
                if loader is not None:
                    return loader
            except Exception as e:
                print(f"[SleepCL] Buffer loader failed ({e}) — no replay data.")

        return None

    def _build_per_task_sleep_loaders(self):
        """
        Build per-task DataLoaders from the internal balanced sleep memory.

        Returns a list of DataLoaders, one per seen task (in task order).
        Each loader only contains that task's data — used for focused rehearsal
        so each old task gets a guaranteed minimum number of consolidation steps.
        Returns None if internal memory is empty.
        """
        if not self._sleep_mem_xs:
            return None
        loaders = []
        for x_task, y_task in zip(self._sleep_mem_xs, self._sleep_mem_ys):
            bs = max(1, min(self.args.batch_size, len(x_task)))
            loaders.append(DataLoader(
                TensorDataset(x_task, y_task),
                batch_size=bs,
                shuffle=True,
            ))
        return loaders

    def _loader_from_buffer(self, buf):
        """
        Extract (X, Y) from various Buffer structures present in this codebase.
        Returns a DataLoader with x in (N, C, T) — channels-first, permuted
        from buffer's (N, T, C) storage so sleep functions can call model(x)
        directly (CNNEncoder no longer transposes internally).
        Returns None if the buffer is empty.
        """
        # Case 1: .x / .y attributes (some custom buffers)
        if hasattr(buf, "x") and hasattr(buf, "y"):
            raw_x = buf.x
            raw_y = buf.y
            if not raw_x:   # empty list
                return None
            X = torch.stack(raw_x) if isinstance(raw_x, list) else raw_x
            Y = torch.stack(raw_y) if isinstance(raw_y, list) else raw_y

        # Case 2: .memory list of (x, y) tuples
        elif hasattr(buf, "memory"):
            if not buf.memory:
                return None
            X = torch.stack([m[0] for m in buf.memory])
            Y = torch.stack([m[1] for m in buf.memory])

        # Case 3: .buffer_input / .buffer_label  (TSCIL standard Buffer)
        elif hasattr(buf, "buffer_input") and hasattr(buf, "buffer_label"):
            bi = buf.buffer_input
            bl = buf.buffer_label
            if isinstance(bi, torch.Tensor):
                if bi.shape[0] == 0:
                    return None
                X = bi
            elif isinstance(bi, list):
                if not bi:
                    return None
                X = torch.stack(bi)
            else:
                return None
            Y = bl if isinstance(bl, torch.Tensor) else torch.stack(bl)

        else:
            return None

        if X.shape[0] == 0:
            return None

        # Buffer stores data as (N, T, C) — permute to (N, C, T) channels-first.
        # CNNEncoder.forward no longer transposes, so model expects channels-first.
        if X.dim() == 3:
            X = X.permute(0, 2, 1).contiguous()

        return DataLoader(
            TensorDataset(X, Y),
            batch_size=self.args.batch_size,
            shuffle=True,
        )

    # ------------------------------------------------------------------
    # Plasticity measurement (before / after sleep comparison)
    # ------------------------------------------------------------------

    def _plasticity_snapshot(self, x_train, y_train, n_samples=64):
        """
        Compute four plasticity indicators on a small batch from the current
        task's training data (x_train is (N,T,C), permuted to (N,C,T) here).

        Returns
        -------
        dict with keys:
          w_norm   – total L2 weight norm (↓ after downregulation)
          eff_rank – entropy-based effective rank of features (↑ after sleep)
          dormant  – fraction of near-zero feature dims (↓ after sleep)
          g_norm   – gradient norm on this task's data (↓ = more settled)
        """
        x_np = x_train.cpu().numpy() if isinstance(x_train, torch.Tensor) else np.asarray(x_train)
        y_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)
        # Same object_ guard as _collect_sleep_memory (Chapman Time-IL).
        if x_np.dtype == object:
            x_np = np.array(list(x_np), dtype=np.float32)

        n = min(n_samples, len(x_np))
        idx = np.random.choice(len(x_np), n, replace=False)
        # x_np is (N,T,C) — permute to (N,C,T) for direct model call
        x_t = torch.FloatTensor(x_np[idx]).permute(0, 2, 1).to(self.device)
        y_t = torch.LongTensor(y_np[idx]).to(self.device)

        # Weight norm — most direct signal of downregulation
        w_norm = sum(p.norm().item() for p in self.model.parameters())

        self.model.eval()
        with torch.no_grad():
            feats = self.model.feature(x_t)  # (N, D)

        # Effective rank (entropy of eigenvalue spectrum)
        X = feats - feats.mean(0, keepdim=True)
        C = (X.T @ X) / max(X.shape[0], 1)
        eps = 1e-6
        C = C + eps * torch.eye(C.shape[0], device=C.device)
        eig = torch.linalg.eigvalsh(C).clamp(min=eps)
        p = eig / eig.sum()
        eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()

        # Dormant ratio: neurons whose mean |activation| < 1% of the max.
        # Quantile-based threshold is self-fulfilling (always ~5%) so we use
        # an absolute threshold relative to the feature energy distribution.
        energy = torch.abs(feats).mean(0)
        thr = energy.max() * 0.01          # < 1% of peak energy = dormant
        dormant = (energy <= thr).float().mean().item()

        # Gradient norm on random noise — measures plasticity (gradient flow
        # capacity) without confounding with retention of current-task data.
        # High g_norm_plastic = model can still update freely = more plastic.
        C_in, T_in = x_t.shape[1], x_t.shape[2]
        x_rand = torch.randn(n, C_in, T_in, device=self.device)
        n_cls = int(y_t.max().item()) + 1
        y_rand = torch.randint(0, n_cls, (n,), device=self.device)

        self.model.train()
        self.model.zero_grad()
        out_rand = self.model(x_rand)
        F.cross_entropy(out_rand, y_rand).backward()
        g_norm_plastic = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
        self.model.zero_grad()

        # Retention gradient norm — how settled the model is on current task.
        self.model.zero_grad()
        out = self.model(x_t)
        F.cross_entropy(out, y_t).backward()
        g_norm_retain = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
        self.model.zero_grad()

        return {
            'w_norm':        w_norm,
            'eff_rank':      eff_rank,
            'dormant':       dormant,
            'g_plastic':     g_norm_plastic,   # ↑ after sleep = more plastic
            'g_retain':      g_norm_retain,    # ↓ after sleep = well-settled
        }

    def _log_sleep_plasticity_delta(self, pre, post, task_id):
        """Print a before/after sleep comparison table and accumulate to log."""
        labels = {
            'w_norm':    'weight_norm  (↓ = downreg works)',
            'eff_rank':  'eff_rank     (↑ = repr spread)',
            'dormant':   'dormant      (↓ = reactivation)',
            'g_plastic': 'g_plastic    (↑ = more plastic)',
            'g_retain':  'g_retain     (↓ = well-settled)',
        }
        delta     = {k: post[k] - pre[k] for k in pre}
        pct_delta = {k: delta[k] / (abs(pre[k]) + 1e-9) * 100 for k in pre}

        print(f"\n[SleepCL] Task {task_id} — sleep plasticity delta:")
        print(f"  {'metric':<34s} {'before':>10s} {'after':>10s} {'Δ':>10s} {'%Δ':>8s}")
        print(f"  {'-'*34} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for k, label in labels.items():
            print(f"  {label:<34s} {pre[k]:10.4f} {post[k]:10.4f} {delta[k]:+10.4f} {pct_delta[k]:+7.1f}%")
        print()

        # Accumulate for result.pkl — stored as plain Python floats (not tensors)
        self._sleep_plasticity_log.append({
            'task_id':   task_id,
            'pre':       {k: float(v) for k, v in pre.items()},
            'post':      {k: float(v) for k, v in post.items()},
            'delta':     {k: float(v) for k, v in delta.items()},
            'pct_delta': {k: float(v) for k, v in pct_delta.items()},
        })



    def _get_importance_for_sleep(self):
        """
        Extract parameter importance from the base CL agent (if available)

        Returns (importance_dict, source_name) or (None, 'None').
        """
        import torch.nn.functional as F_

        # 1. EWC: self.importances = {task_id: [(name, tensor), ...]}
        if hasattr(self, 'importances') and self.importances:
            merged = {}
            for task_imps in self.importances.values():
                for name, imp in task_imps:
                    imp_dev = imp.to(self.device)
                    if name not in merged:
                        merged[name] = imp_dev.clone()
                    else:
                        a, b = merged[name], imp_dev
                        if a.shape == b.shape:
                            merged[name] = torch.max(a, b)
                        else:
                            # Growing head: pad smaller tensor to match larger
                            target = tuple(max(sa, sb)
                                           for sa, sb in zip(a.shape, b.shape))
                            def _pad(t, tgt):
                                spec = []
                                for d in reversed(range(t.dim())):
                                    spec.extend([0, tgt[d] - t.shape[d]])
                                return F.pad(t, spec, value=0.0)
                            merged[name] = torch.max(_pad(a, target), _pad(b, target))
            return merged, 'EWC_Fisher'

        # 2. MAS: self.importance = {name: tensor}
        if hasattr(self, 'importance') and isinstance(self.importance, dict) \
                and len(self.importance) > 0:
            return {k: v.to(self.device) for k, v in self.importance.items()}, 'MAS_Omega'

        # 3. SI: self.ewc_data = (theta_star, omega)
        #    ewc_data[1] holds flat importance vectors; reshape to param shapes.
        if hasattr(self, 'ewc_data') and self.ewc_data[1]:
            importance = {}
            for name, p in self.model.named_parameters():
                if name in self.ewc_data[1]:
                    flat = self.ewc_data[1][name]
                    try:
                        importance[name] = flat.reshape(p.shape).to(self.device)
                    except RuntimeError:
                        importance[name] = flat[:p.numel()].reshape(p.shape).to(self.device)
            if importance:
                return importance, 'SI_PathIntegral'

        # 4. DER: buffer stores soft logits → compute soft-target Fisher
        buf = getattr(self, 'buffer', None)
        if buf is not None and hasattr(buf, 'buffer_logits') \
                and buf.buffer_logits is not None:
            n_valid = getattr(buf, 'current_index', 0)
            if n_valid > 0:
                X_buf = buf.buffer_input[:n_valid]     # (N, T, C)
                L_buf = buf.buffer_logits[:n_valid]    # (N, n_cls)
                fisher = {n: torch.zeros_like(p)
                          for n, p in self.model.named_parameters()
                          if p.requires_grad}
                self.model.eval()
                bs = 64
                n_batches = min(5, max(1, n_valid // bs))
                perm = torch.randperm(n_valid)[:n_batches * bs]
                for i in range(n_batches):
                    idx = perm[i * bs:(i + 1) * bs]
                    xb = X_buf[idx].permute(0, 2, 1).to(self.device)  # (N,T,C)→(N,C,T)
                    lb = L_buf[idx].to(self.device)
                    T_ = 2.0
                    out = self.model(xb)
                    # Align to narrower of model output vs stored logits (growing head)
                    n_cls = min(out.size(1), lb.size(1))
                    loss = F_.kl_div(
                        F_.log_softmax(out[:, :n_cls] / T_, dim=1),
                        F_.softmax(lb[:, :n_cls]  / T_, dim=1),
                        reduction='batchmean',
                    ) * T_ ** 2
                    self.model.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        for n, p in self.model.named_parameters():
                            if p.grad is not None:
                                fisher[n] += p.grad.pow(2)
                self.model.zero_grad()
                for n in fisher:
                    fisher[n] /= n_batches
                return fisher, 'DER_SoftFisher'

        # 5. Fallback: ad-hoc diagonal Fisher from sleep memory
        loader = self._build_sleep_replay_loader()
        if loader is not None:
            importance = compute_fisher_diagonal(
                self.model, loader, n_batches=5, device=self.device)
            return importance, 'AdHoc_Fisher'

        return None, 'None'

    def _do_sleep_consolidation(self, replay_loader, task_id):
        """
        Run sleep objective optimisation and merge result into model.
        """
        input_shape = self._infer_input_shape(replay_loader)
        num_classes  = self._infer_num_classes(replay_loader)
        theta_prev   = getattr(self, '_theta_prev', None)
        per_task_loaders = self._build_per_task_sleep_loaders()

        # ── Step 1: importance for sleep regularisation ─────────
        importance_dict, imp_source = self._get_importance_for_sleep()
        print(f"[SleepCL] task={task_id} | importance source: {imp_source}")

        # ── Step 2: inject g_plastic / g_retain into cfg for adaptive lr ─
        if self._sleep_plasticity_log:
            last_pre = self._sleep_plasticity_log[-1]['pre']
            self.sleep_obj_cfg._g_plastic_prev = last_pre.get('g_plastic', 1.0)
            self.sleep_obj_cfg._g_retain_prev  = last_pre.get('g_retain',  1.0)

        # ── Step 3: 4-phase oscillation sleep ────────────────────────────
        theta_final = compute_theta_final_sleep_objective(
            model_wake=self.model,
            memory_loader=replay_loader,
            bank=self.bank,
            task_id=task_id,
            input_shape=input_shape,
            num_classes=num_classes,
            device=self.device,
            cfg=self.sleep_obj_cfg,
            theta_prev=theta_prev,
            per_task_loaders=per_task_loaders,
            importance_dict=importance_dict,   # 방안 1
        )

        # ── Step 4: adaptive merge_lam ──────────────────────

        import copy as _copy
        if getattr(self.sleep_obj_cfg, 'adaptive_merge', True):
            _model_sleep_tmp = _copy.deepcopy(self.model)
            _model_sleep_tmp.load_state_dict(theta_final, strict=True)
            _teacher_wake_tmp = _copy.deepcopy(self.model)
            effective_lam = compute_adaptive_merge_lam(
                model_sleep=_model_sleep_tmp,
                teacher_wake=_teacher_wake_tmp,
                memory_loader=replay_loader,
                device=self.device,
                base_lam=self.sleep_merge_lam,
                quality_weight=getattr(self.sleep_obj_cfg, 'merge_quality_weight', 0.5),
            )
            del _model_sleep_tmp, _teacher_wake_tmp
        else:
            effective_lam = self.sleep_merge_lam

        # θ ← (1-λ)·θ_wake + λ·θ_sleep
        merge_theta_wake_and_final(self.model, theta_final, lam_merge=effective_lam)
        print(f"[SleepCL] task={task_id} | consolidation done "
              f"(merge_lam={effective_lam:.3f}, imp={imp_source})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_input_shape(self, loader):
        x, _ = next(iter(loader))
        return tuple(x.shape[1:])

    def _infer_num_classes(self, loader):
        ys = []
        for _, y in loader:
            ys.append(y)
            if len(ys) >= 3:
                break
        return int(torch.cat(ys).max().item()) + 1


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def make_sleep_cl_agent(model, args):
    """
    Factory that creates a SleepCL agent dynamically by combining
    SleepCLMixin with the base agent specified by args.sleep_base_agent.

    Called from agents/utils/name_match.py as:
        agents['sleep-CL'] = make_sleep_cl_agent
    and invoked by exp.py as:
        agent = agents['sleep-CL'](model=model, args=args)
    """
    from agents.utils.name_match import _BASE_AGENTS_FOR_SLEEP

    base_name = getattr(args, "sleep_base_agent", "SFT")
    if base_name not in _BASE_AGENTS_FOR_SLEEP:
        raise ValueError(
            f"Unknown sleep_base_agent '{base_name}'. "
            f"Choose from: {list(_BASE_AGENTS_FOR_SLEEP.keys())}"
        )

    base_cls = _BASE_AGENTS_FOR_SLEEP[base_name]
    CombinedCls = type(f"SleepCL_{base_name}", (SleepCLMixin, base_cls), {})
    return CombinedCls(model, args)
