#!/usr/bin/env python3
"""
optimizer_scaling.py — Optimizer Scaling Comparison Study (Figure 1)

Compares 5 optimizers (3 gradient-free, 2 gradient-based) on synthetic
phantom recovery at INCREASING problem sizes to show how gradient-free
methods degrade as dimensionality grows.

Scaling: runs at 1, 5, 10, 25, 50, 100, 250, 500 A-lines
(~40 to ~20,000 tissue pixels).  Each scale uses the same forward model,
initial conditions, and iteration budget.  All gradient-free methods
evaluate 20 candidates per iteration for a fair function-evaluation budget.

Optimizers tested:
  1. Adam (gradient-based, Kingma & Ba 2014)
  2. L-BFGS-B (gradient-based, bound-constrained quasi-Newton,
     Byrd, Lu, Nocedal & Zhu 1995)
  3. Random Search (gradient-free, Solis & Wets 1981 Basic Algorithm
     with Algorithm 1 sampling rule)
  4. CMA-ES (gradient-free, Hansen & Ostermeier 2001, via pycma with
     CMA_active=False and updatecovwait=⌈√n⌉)

Output: outputs/optimizer_scaling/
  - results.json
  - summary_table.tex (LaTeX)
  - scaling_plot.png

Usage:
    python experiments/optimizer_scaling.py
"""

import math
import os
import sys
import time
import json

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import minimize as scipy_minimize
import cma  # pycma: reference implementation by N. Hansen

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── path setup ─────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.forward import create_forward_model
from src.inverse import compute_loss, build_noise_floor_mask

# ── constants (matching 0b_inverse_model.py) ───────────────────────────────
_SHARED_ROOT = "/home/gridsan/amanicka/erelab_shared/Ajay/DiffOCT-Phantom-Experiments"
_LOCAL_ROOT = PHANTOM_ROOT
_PHANTOM_SUBDIR = "outputs/inverse_model_verification/phantom_0000"
PHANTOM_DATA_DIR = os.path.join(_LOCAL_ROOT, _PHANTOM_SUBDIR)
if not os.path.isdir(PHANTOM_DATA_DIR):
    # Fall back to the shared copy, then defer the hard failure to
    # load_phantom_data(). Keeps this module importable from a worktree with
    # no local outputs/ so axial_scaling_study.py can reuse the optimizer
    # implementations below without re-running the phantom generator.
    _shared_dir = os.path.join(_SHARED_ROOT, _PHANTOM_SUBDIR)
    if os.path.isdir(_shared_dir):
        PHANTOM_DATA_DIR = _shared_dir

_OUT_SUFFIX = os.environ.get('OUT_SUFFIX', '')
OUT_DIR = os.path.join(PHANTOM_ROOT, "outputs", f"optimizer_scaling{_OUT_SUFFIX}")

NUM_SEEDS = 8              # one seed per GPU in the per-seed parallel rerun
NUM_ITERS = int(os.environ.get('NUM_ITERS', 20000))
# 18 scales — matches what figureS1a_bscan_scaling.ipynb expects.
ALINE_SCALES = [1, 2, 3, 5, 10, 15, 25, 50, 75, 100,
                150, 200, 250, 300, 350, 400, 450, 500]
MU_AIR = 1e-4
MU_MIN = 1e-2              # same as 0b opt_config
MU_MAX = 20.0              # same as 0b opt_config
LR = 1e-2                  # higher LR for 5k iterations
LOSS_SCALE = 1e3
MAX_DELTA_MU_MM = 1.0      # gradient clamp: ±MAX_DELTA_MU_MM/LR
NOISE_FLOOR_DB = 79.0
LOG_INTERVAL = 100

OPTIMIZER_ORDER = [
    'Adam',
    'L-BFGS',
    'Random Search',
    'CMA-ES',
]

GENERATE_CONV_PLOT = os.environ.get('GENERATE_CONV_PLOT', '0') == '1'
# Sharded GPU mode: SCALE_INDEX selects a single ALINE_SCALES entry and
# writes results to a per-scale JSON shard for later aggregation.
# MERGE_CONV=1 aggregates convergence-study shards; MERGE_BASELINE=1
# aggregates scaling-study shards. Both dispatch to their merge function.
SCALE_INDEX = os.environ.get('SCALE_INDEX', '')
SEED_INDEX  = os.environ.get('SEED_INDEX', '')
MERGE_CONV = os.environ.get('MERGE_CONV', '0') == '1'
MERGE_BASELINE = os.environ.get('MERGE_BASELINE', '0') == '1'
MERGE_PER_SEED = os.environ.get('MERGE_PER_SEED', '0') == '1'


# ═══════════════════════════════════════════════════════════════════════════
# Data loading and setup
# ═══════════════════════════════════════════════════════════════════════════

def load_phantom_data(device):
    """Load ground truth mu, alpha, target, and frozen gain from saved phantom."""
    if not os.path.isdir(PHANTOM_DATA_DIR):
        raise FileNotFoundError(
            f"Cannot find phantom data at {PHANTOM_DATA_DIR}. "
            f"Run experiments/inverse_model_verification.py first.")
    mu_gt_np = np.load(os.path.join(PHANTOM_DATA_DIR, 'mu_gt_mm.npy'))
    alpha_gt_np = np.load(os.path.join(PHANTOM_DATA_DIR, 'alpha_gt.npy'))
    target_dB_np = np.load(os.path.join(PHANTOM_DATA_DIR, 'target_dB.npy'))

    with open(os.path.join(PHANTOM_DATA_DIR, 'results.json')) as f:
        phantom_results = json.load(f)
    gain_dB_val = phantom_results['gain_dB_frozen']

    mu_gt = torch.tensor(mu_gt_np, dtype=torch.float32, device=device)
    alpha_gt = torch.tensor(alpha_gt_np, dtype=torch.float32, device=device)
    target_dB = torch.tensor(target_dB_np, dtype=torch.float32, device=device)

    return mu_gt, alpha_gt, target_dB, mu_gt_np, gain_dB_val


def generate_initial_conditions(mu_gt_np, num_seeds, device):
    """Generate num_seeds deterministic random initial mu maps in [0, 10]."""
    tissue_mask = (mu_gt_np > MU_AIR * 10).astype(np.float32)
    inits = []
    for seed in range(num_seeds):
        rng = np.random.default_rng(seed=42 + seed)
        mu_scrambled = rng.uniform(0.0, 10.0,
                                   size=mu_gt_np.shape).astype(np.float32)
        mu_init = mu_scrambled * tissue_mask + MU_AIR * (1 - tissue_mask)
        inits.append(torch.tensor(mu_init, dtype=torch.float32, device=device))
    return inits


def eval_loss_nograph(forward_model, mu, alpha, target, noise_mask):
    """Evaluate forward model loss without autograd (for gradient-free)."""
    with torch.no_grad():
        pred = forward_model(mu, alpha, noise_mask=noise_mask)
        loss = compute_loss(pred, target, noise_mask, LOSS_SCALE)
    return loss.item()


def compute_raw_mse(forward_model, mu, alpha, target, noise_mask):
    """Compute raw (unscaled) MSE in dB^2, matching optimize() diagnostics."""
    eps = 1e-12
    with torch.no_grad():
        pred = forward_model(mu, alpha, noise_mask=noise_mask)
        diff = (pred - target) ** 2
        if noise_mask is not None:
            mse = (diff * noise_mask).sum() / (noise_mask.sum() + eps)
        else:
            mse = diff.mean()
    return mse.item()


def compute_mape(mu_recovered, mu_gt, tissue_bool):
    """MAPE (%) on tissue pixels — same as 0b_inverse_model.py."""
    with torch.no_grad():
        mu_rec = mu_recovered[tissue_bool].cpu().numpy()
        mu_ref = mu_gt[tissue_bool].cpu().numpy()
        abs_err = np.abs(mu_rec - mu_ref)
        rel_err = abs_err / (mu_ref + 1e-12)
    return float(rel_err.mean() * 100)


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer implementations
# ═══════════════════════════════════════════════════════════════════════════

def run_adam(forward_model, mu_init, alpha, target, noise_mask, device,
            num_iters, seed_idx=0):
    """Adam (Kingma & Ba 2014, Algorithm 1) + PGD onto [MU_MIN, MU_MAX] and
    the noise-floor mask. No gradient clipping or nan sanitization —
    gradients flow into torch.optim.Adam unmodified.
    """
    mu = torch.nn.Parameter(mu_init.clone())
    # alpha as frozen Parameter (matches optimize() which puts it in Adam)
    alpha_param = torch.nn.Parameter(alpha.clone(), requires_grad=False)
    optimizer = torch.optim.Adam([mu, alpha_param], lr=LR)

    # Zero noise-floor pixels at init
    with torch.no_grad():
        mu.data *= noise_mask

    mse_history = []

    # Best-model tracking (uses training loss — no extra forward pass)
    best_mse = float('inf')
    best_mu = mu.detach().clone()

    # CUDA kernel warmup: one untimed forward+backward, no optimizer step,
    # so no parameter update happens before the timed loop.
    pred = forward_model(mu, alpha_param, noise_mask=noise_mask)
    loss = compute_loss(pred, target, noise_mask, LOSS_SCALE)
    loss.backward()
    optimizer.zero_grad(set_to_none=False)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    fwd_evals = 0  # count forward_model() calls inside the timed region

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=False)
        pred = forward_model(mu, alpha_param, noise_mask=noise_mask)
        fwd_evals += 1
        loss = compute_loss(pred, target, noise_mask, LOSS_SCALE)

        # Track best model using current loss (pre-step, no extra fwd pass)
        mse_val = loss.item() / LOSS_SCALE
        if mse_val < best_mse:
            best_mse = mse_val
            best_mu.copy_(mu.detach())

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mu.data.clamp_(MU_MIN, MU_MAX)
            mu.data.mul_(noise_mask)

        if it % LOG_INTERVAL == 0 or it == num_iters - 1:
            mse_history.append((it, mse_val))

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_wall_sec = time.perf_counter() - t_start
    avg_iter_ms = (total_wall_sec / num_iters) * 1000
    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if device.type == 'cuda' else 0.0)

    # Restore best model (matches optimize())
    mu.data.copy_(best_mu)
    return (mu.detach(), avg_iter_ms, mse_history, fwd_evals,
            total_wall_sec, peak_mem_mb)


def run_lbfgs(forward_model, mu_init, alpha, target, noise_mask, device,
              num_iters, seed_idx=0):
    """L-BFGS-B (Byrd, Lu, Nocedal & Zhu 1995, "A Limited Memory Algorithm
    for Bound Constrained Optimization", SIAM J. Sci. Comput. 16(5):
    1190-1208) via scipy.optimize.minimize(method='L-BFGS-B'), the
    reference implementation wrapping the Fortran routine L-BFGS-B.f by
    Nocedal and Zhu.

    L-BFGS-B is the bound-constrained variant of L-BFGS (Nocedal & Wright
    2006 Alg. 7.5, Byrd et al. paper). It uses the same two-loop
    recursion (Alg. 7.4) and γ_k scaling (eq. 7.20) for the Hessian
    approximation, but handles box constraints μ ∈ [MU_MIN, MU_MAX]
    natively via gradient projection and subspace minimization — rather
    than the naive post-step PGD projection that would corrupt curvature
    pairs (s_k, y_k) in unconstrained L-BFGS. Strong Wolfe line search is
    performed on the feasible subspace.

    Each function evaluation computes both f(x) and ∇f(x) from a single
    forward+backward pass on the GPU, returned to scipy as (scalar,
    numpy-array) via jac=True. The GPU does the compute; scipy handles
    curvature storage and line search on CPU (cheap O(m·n) ops).

    Settings for strict replication:
      - maxiter = num_iters: outer L-BFGS iteration budget
      - maxcor = 10: history size m (pycma/paper default, in [3, 20])
      - maxls = 20 (default): line search steps per iter
      - ftol = gtol = 0: termination strictly by maxiter
      - bounds = [MU_MIN, MU_MAX] per tissue pixel

    Non-tissue pixels are not passed to L-BFGS-B; they remain zero in the
    reconstructed μ via the mask subspace.
    """
    tissue_idx = torch.nonzero(noise_mask.flatten() > 0.5,
                               as_tuple=False).squeeze(1)
    n_tissue = int(tissue_idx.numel())

    mu_init_masked = mu_init.clone()
    mu_init_masked.mul_(noise_mask)
    x0_np = mu_init_masked.view(-1)[tissue_idx].detach().cpu().numpy().astype(np.float64)

    # Work tensor with autograd enabled; only tissue pixels are written
    mu_work = torch.nn.Parameter(mu_init_masked.clone(), requires_grad=True)

    call_count = [0]
    best = {'loss': float('inf'), 'x': x0_np.copy()}
    mse_history = []

    def fun_and_grad(x_np):
        """Scipy callback: return (loss, gradient) as (float, ndarray)."""
        x_tensor = torch.from_numpy(x_np.astype(np.float32)).to(device)
        with torch.no_grad():
            mu_work.data.view(-1)[tissue_idx] = x_tensor
        if mu_work.grad is not None:
            mu_work.grad.zero_()
        pred = forward_model(mu_work, alpha, noise_mask=noise_mask)
        loss = compute_loss(pred, target, noise_mask, LOSS_SCALE)
        loss.backward()
        loss_val = float(loss.item())
        grad_tissue = (mu_work.grad.view(-1)[tissue_idx]
                       .detach().cpu().numpy().astype(np.float64))
        if loss_val < best['loss']:
            best['loss'] = loss_val
            best['x'] = x_np.copy()
        if (call_count[0] % LOG_INTERVAL == 0
                or call_count[0] == num_iters - 1):
            mse_history.append((call_count[0], best['loss'] / LOSS_SCALE))
        call_count[0] += 1
        return loss_val, grad_tissue

    bounds = [(MU_MIN, MU_MAX)] * n_tissue

    # CUDA kernel warmup: one untimed forward+backward, no scipy update.
    with torch.no_grad():
        _ = forward_model(mu_work, alpha, noise_mask=noise_mask)
    if mu_work.grad is not None:
        mu_work.grad.zero_()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    scipy_minimize(
        fun_and_grad,
        x0=x0_np,
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': num_iters,
            'maxfun': num_iters * 20,  # loose cap; maxiter is the real budget
            'ftol': 0.0,
            'gtol': 0.0,
            'maxcor': 10,
        },
    )

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_wall_sec = time.perf_counter() - t_start

    actual_evals = max(call_count[0], 1)
    avg_iter_ms = (total_wall_sec / actual_evals) * 1000
    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if device.type == 'cuda' else 0.0)

    with torch.no_grad():
        best_mu = mu_init_masked.clone()
        best_mu.view(-1)[tissue_idx] = torch.from_numpy(
            best['x'].astype(np.float32)).to(device)

    if not mse_history or mse_history[-1][0] != actual_evals - 1:
        mse_history.append((actual_evals - 1, best['loss'] / LOSS_SCALE))

    return (best_mu, avg_iter_ms, mse_history, call_count[0],
            total_wall_sec, peak_mem_mb)


def run_random_search(forward_model, mu_init, alpha, target, noise_mask,
                      device, num_iters, seed_idx=0):
    """Solis & Wets 1981 'Basic Algorithm' (p. 25) with Algorithm 1
    sampling rule: μ_k = N(x_k + b_k, ρ_k · I).

    Strict replication:
      - single candidate ξ per iteration (full-dim Gaussian, NOT sparse)
      - bias vector b_k with paper's Step 2 update rules
      - reversal branch: try 2·x_k − ξ if ξ fails
      - adaptive ρ: expand by ex=2 after s_ex=5 consecutive successes,
        contract by ct=0.5 after f_ct=3 consecutive failures
      - ρ_{-1}=1, ρ_lb=0 (the paper explicitly permits 0; with a fixed
        iteration budget we don't need early termination)

    Bound-constrained adaptation (paper is S = R^n): ξ and its reversal
    are projected onto [MU_MIN, MU_MAX] × (tissue-mask subspace) before
    evaluation and acceptance. ε is sampled on the tissue subspace so
    the bias stays in-subspace. Bias update uses (ξ − x_k) in the
    unprojected proposal, matching the paper's formulas.

    Expected function evaluations per iter: 1 (accept) or 2 (reversal
    branch). For a fixed num_iters budget this gives ≤2·num_iters total
    forward evaluations, vs ≤num_iters for gradient-based methods.
    """
    # Paper constants (Basic Algorithm, p. 25)
    S_EX = 5
    F_CT = 3
    EX_FAC = 2.0
    CT_FAC = 0.5
    RHO_INIT = 1.0
    RHO_LB = 0.0

    mu = mu_init.clone()
    mu.mul_(noise_mask)

    rho = RHO_INIT
    bias = torch.zeros_like(mu)
    num_s = 0
    num_f = 0

    current_loss = eval_loss_nograph(forward_model, mu, alpha, target, noise_mask)
    best_loss = current_loss
    best_mu = mu.clone()
    mse_history = []

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    fwd_evals = 0  # counts eval_loss_nograph calls inside the timed region

    for it in range(num_iters):
        # Step 1: ρ update per S&W 1981 Algorithm 1 (p. 25). Counters are
        # reset only in Step 2 based on iteration outcome, not at Step 1
        # — the literal formula allows ρ to keep expanding/contracting
        # while a success/failure streak persists, which is the paper's
        # intended behavior.
        if num_s >= S_EX:
            rho = rho * EX_FAC
        elif num_f >= F_CT:
            rho = rho * CT_FAC
        if rho <= RHO_LB:
            break

        # ξ ~ N(x_k + b_k, ρ² · I), restricted to tissue subspace
        eps = torch.randn_like(mu) * noise_mask
        xi = mu + bias + rho * eps
        xi_proj = xi.clamp(MU_MIN, MU_MAX) * noise_mask
        loss_xi = eval_loss_nograph(forward_model, xi_proj, alpha, target,
                                    noise_mask)
        fwd_evals += 1

        # Step 2: accept / reversal / reject
        if loss_xi < current_loss:
            bias = 0.4 * (xi - mu) + 0.2 * bias
            mu = xi_proj
            current_loss = loss_xi
            num_s += 1
            num_f = 0
        else:
            rev = 2.0 * mu - xi
            rev_proj = rev.clamp(MU_MIN, MU_MAX) * noise_mask
            loss_rev = eval_loss_nograph(forward_model, rev_proj, alpha,
                                         target, noise_mask)
            fwd_evals += 1
            if loss_rev < current_loss:
                bias = bias - 0.4 * (xi - mu)
                mu = rev_proj
                current_loss = loss_rev
                num_s += 1
                num_f = 0
            else:
                bias = 0.5 * bias
                num_s = 0
                num_f += 1

        if current_loss < best_loss:
            best_loss = current_loss
            best_mu = mu.clone()

        if it % LOG_INTERVAL == 0 or it == num_iters - 1:
            mse_history.append((it, best_loss / LOSS_SCALE))

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_wall_sec = time.perf_counter() - t_start
    avg_iter_ms = (total_wall_sec / num_iters) * 1000
    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if device.type == 'cuda' else 0.0)

    return (best_mu, avg_iter_ms, mse_history, fwd_evals,
            total_wall_sec, peak_mem_mb)



def run_cma_es(forward_model, mu_init, alpha, target, noise_mask,
               device, num_iters, seed_idx=0):
    """(μ_W, λ)-CMA-ES from Hansen & Ostermeier 2001 ("Completely
    Derandomized Self-Adaptation in Evolution Strategies", Evolutionary
    Computation 9(2):159-195), Section 5 / Appendix A. Implemented via
    pycma — the reference implementation maintained by N. Hansen, the
    paper's first author.

    Settings for strict 2001-paper replication (atop pycma 3.x):
      - CMA_active=False: disables the Jastrebski & Arnold 2006 active
        extension (negative updates from worst individuals); the paper
        we cite predates this extension
      - full (non-diagonal) covariance matrix throughout (pycma default)
      - default λ = 4 + ⌊3·ln(n)⌋, μ = ⌊λ/2⌋, weights wᵢ = ln((λ+1)/2)−ln(i)
      - default cumulation/damping: cc = 4/(n+4), ccov = 2/(n+√2)²,
        cσ = 4/(n+4), dσ = 1/cσ + 1 (pycma defaults match Table 1)
      - initial step size σ₀ = 1.0, C⁽⁰⁾ = I
      - tolfun = tolx = 0: termination strictly by maxfevals budget
      - updatecovwait = ⌈√n⌉: lazy eigendecomposition of C, as the
        paper explicitly recommends in §5.2 "Practical Hints" (p. 178):
        "it is possible to update B⁽ᵍ⁾ and D⁽ᵍ⁾ not after every
        generation but after √n or even after, e.g., n/10 generations.
        This reduces the computational effort of the strategy from
        O(n³) to O(n^{2.5}) or O(n²), respectively." We use √n so that
        the eigendecomposition fires multiple times even at the largest
        problem size (n = 20 000 has 606 generations in our 2×10⁴-eval
        budget; √n = 141 gives ~4 decompositions per run vs zero under
        the n/10 setting)

    Bound-constrained adaptation via pycma's built-in bound handler
    (projection onto [MU_MIN, MU_MAX]). Non-tissue pixels are not passed
    to CMA-ES; they remain zero via the mask subspace. Budget capped at
    num_iters objective evaluations via maxfevals.

    Budget enforcement: when the outer eval counter reaches num_iters
    mid-generation, we break the ask/tell loop BEFORE calling es.tell()
    on the partial population. This prevents a truncated fitness vector
    from corrupting CMA's covariance update. The discarded generation's
    evaluations are still counted in fwd_evals; the best-μ tracked in
    the objective callback is unaffected.
    """
    tissue_idx = torch.nonzero(noise_mask.flatten() > 0.5,
                               as_tuple=False).squeeze(1)
    n_tissue = int(tissue_idx.numel())

    mu_init_masked = mu_init.clone()
    mu_init_masked.mul_(noise_mask)
    x0 = mu_init_masked.view(-1)[tissue_idx].detach().cpu().numpy().astype(np.float64)
    sigma0 = 1.0  # paper's default initial step size

    mu_work = mu_init_masked.clone()

    call_count = [0]
    best = {'loss': float('inf'),
            'x': x0.copy()}
    mse_history = []

    def objective(x_np):
        x_arr = np.asarray(x_np, dtype=np.float32)
        x_tensor = torch.from_numpy(x_arr).to(device)
        with torch.no_grad():
            mu_work.view(-1)[tissue_idx] = x_tensor
        loss = eval_loss_nograph(forward_model, mu_work, alpha, target,
                                 noise_mask)
        if loss < best['loss']:
            best['loss'] = loss
            best['x'] = np.asarray(x_np, dtype=np.float64).copy()
        if (call_count[0] % LOG_INTERVAL == 0
                or call_count[0] == num_iters - 1):
            mse_history.append((call_count[0], best['loss'] / LOSS_SCALE))
        call_count[0] += 1
        return loss

    opts = {
        'bounds': [[MU_MIN] * n_tissue, [MU_MAX] * n_tissue],
        'maxfevals': num_iters,
        'seed': seed_idx + 1,      # pycma: 0 is interpreted as 'time'; +1 so seed_idx=0 maps to a valid non-time seed
        'verbose': -9,
        'tolfun': 0.0,
        'tolx': 0.0,
        'CMA_active': False,       # strict H&O 2001, no Jastrebski-Arnold
        'updatecovwait': max(1, int(math.ceil(math.sqrt(n_tissue)))),  # H&O 2001 §5.2 √n option
    }

    # CUDA kernel warmup: one untimed forward pass.
    with torch.no_grad():
        _ = forward_model(mu_work, alpha, noise_mask=noise_mask)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    try:
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
        while not es.stop() and call_count[0] < num_iters:
            solutions = es.ask()
            fitnesses = [objective(x) for x in solutions]
            if call_count[0] >= num_iters:
                # maxfevals reached mid-generation; skip tell to avoid
                # mixing partial populations
                break
            es.tell(solutions, fitnesses)
    except (MemoryError, np.linalg.LinAlgError) as e:
        print(f"  [CMA-ES] terminated early (n={n_tissue}): {type(e).__name__}: {e}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_wall_sec = time.perf_counter() - t_start

    actual_evals = max(call_count[0], 1)
    avg_iter_ms = (total_wall_sec / actual_evals) * 1000
    peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if device.type == 'cuda' else 0.0)

    with torch.no_grad():
        best_mu = mu_init_masked.clone()
        best_mu.view(-1)[tissue_idx] = torch.from_numpy(
            best['x'].astype(np.float32)).to(device)

    if not mse_history or mse_history[-1][0] != actual_evals - 1:
        mse_history.append((actual_evals - 1, best['loss'] / LOSS_SCALE))

    return (best_mu, avg_iter_ms, mse_history, call_count[0],
            total_wall_sec, peak_mem_mb)


def run_nelder_mead(forward_model, mu_init, alpha, target, noise_mask,
                    device, num_iters):
    """Nelder-Mead (simplex) via scipy.optimize.minimize on tissue pixels only.

    Operates only on the tissue pixel subset to reduce dimensionality.
    Limited to num_iters function evaluations (maxfev).
    """
    from scipy.optimize import minimize

    mu = mu_init.clone()
    mu.mul_(noise_mask)

    tissue_bool = noise_mask > 0.5
    n_tissue = int(tissue_bool.sum().item())

    # Extract tissue pixel values as the optimization vector
    x0 = mu[tissue_bool].cpu().numpy().astype(np.float64)

    # Track history and best
    eval_count = [0]
    best_loss = [float('inf')]
    best_x = [x0.copy()]
    mse_history = []

    def objective(x):
        mu_full = mu.clone()
        mu_full[tissue_bool] = torch.tensor(x, dtype=torch.float32,
                                            device=device).clamp(MU_MIN, MU_MAX)
        loss = eval_loss_nograph(forward_model, mu_full, alpha, target,
                                 noise_mask)
        eval_count[0] += 1
        if loss < best_loss[0]:
            best_loss[0] = loss
            best_x[0] = x.copy()
        if eval_count[0] % LOG_INTERVAL == 0:
            mse_history.append((eval_count[0], best_loss[0] / LOSS_SCALE))
        return loss

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.time()

    # Give NM 20x evaluations to match the 20 candidates/iter of other methods
    maxfev = num_iters * 20
    minimize(objective, x0, method='Nelder-Mead',
             options={'maxfev': maxfev, 'maxiter': maxfev,
                      'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.time() - t_start
    n_evals = eval_count[0]
    # Report per-"iteration" time (each iter = 20 evaluations for fairness)
    equiv_iters = max(n_evals / 20, 1)
    avg_iter_ms = (total_time / equiv_iters) * 1000

    # Final entry
    mse_history.append((int(equiv_iters), best_loss[0] / LOSS_SCALE))

    # Reconstruct best mu
    mu_final = mu.clone()
    mu_final[tissue_bool] = torch.tensor(best_x[0], dtype=torch.float32,
                                         device=device).clamp(MU_MIN, MU_MAX)
    return mu_final.detach(), avg_iter_ms, mse_history


# ═══════════════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════════════

OPTIMIZERS = {
    'Adam': run_adam,
    'L-BFGS': run_lbfgs,
    'Random Search': run_random_search,
    'CMA-ES': run_cma_es,
}


def collect_provenance():
    """Capture hardware/software context for reproducibility (reported in
    paper appendix; standard in ICML/CVPR/ICCV reproducibility checklists).
    """
    import platform, socket
    prov = {
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'scipy': __import__('scipy').__version__,
    }
    try:
        prov['cma'] = cma.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        prov['gpu'] = torch.cuda.get_device_name(0)
        prov['cuda'] = torch.version.cuda
        try:
            prov['cudnn'] = torch.backends.cudnn.version()
        except Exception:
            pass
    else:
        prov['gpu'] = 'cpu'
    prov['num_seeds'] = NUM_SEEDS
    prov['num_iters'] = NUM_ITERS
    prov['clock'] = 'time.perf_counter (monotonic)'
    prov['cuda_sync'] = ('torch.cuda.synchronize() before and after each '
                        'timed region; one untimed warmup forward per run')
    prov['peak_mem_protocol'] = ('torch.cuda.reset_peak_memory_stats() '
                                 'before t_start; max_memory_allocated() at end')
    return prov


# ═══════════════════════════════════════════════════════════════════════════
# Main — Scaling study
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Two sharding modes (mutually exclusive):
    #   SCALE_INDEX → one task per ALINE_SCALES entry, all NUM_SEEDS seeds run
    #                inside that task. Writes results_scale{N:04d}.json.
    #   SEED_INDEX  → one task per seed (preferred when len(SCALES) > GPUs):
    #                that single seed runs across every ALINE_SCALES entry.
    #                Writes results_seed{S}.json. Aggregate with MERGE_PER_SEED=1.
    if SCALE_INDEX != '' and SEED_INDEX != '':
        raise SystemExit("Set SCALE_INDEX or SEED_INDEX, not both.")
    scale_shard = SCALE_INDEX != ''
    seed_shard  = SEED_INDEX  != ''

    if scale_shard:
        try:
            shard_idx = int(SCALE_INDEX)
            scales_to_run = [ALINE_SCALES[shard_idx]]
        except (ValueError, IndexError):
            raise SystemExit(f"Invalid SCALE_INDEX={SCALE_INDEX!r}; "
                             f"must be integer in [0, {len(ALINE_SCALES)-1}]")
        seeds_to_run = list(range(NUM_SEEDS))
    elif seed_shard:
        try:
            single_seed = int(SEED_INDEX)
            assert 0 <= single_seed < NUM_SEEDS
        except (ValueError, AssertionError):
            raise SystemExit(f"Invalid SEED_INDEX={SEED_INDEX!r}; "
                             f"must be integer in [0, {NUM_SEEDS-1}]")
        scales_to_run = list(ALINE_SCALES)
        seeds_to_run  = [single_seed]
    else:
        scales_to_run = list(ALINE_SCALES)
        seeds_to_run  = list(range(NUM_SEEDS))

    print("=" * 70)
    print("  Optimizer Scaling Comparison Study")
    if scale_shard:
        print(f"  SCALE shard: SCALE_INDEX={SCALE_INDEX} -> "
              f"{scales_to_run[0]} A-lines only")
    if seed_shard:
        print(f"  SEED shard:  SEED_INDEX={SEED_INDEX} -> seed {single_seed} only")
    print(f"  {len(OPTIMIZER_ORDER)} optimizers x {len(scales_to_run)} scales "
          f"x {len(seeds_to_run)} seeds x {NUM_ITERS:,} iters")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load full phantom data ──────────────────────────────────────────
    print(f"\nLoading phantom data from:\n  {PHANTOM_DATA_DIR}")
    mu_gt_full, alpha_gt_full, target_dB_full, mu_gt_np_full, gain_dB_val = \
        load_phantom_data(device)
    print(f"  Full phantom shape: {mu_gt_np_full.shape}")

    # ── Run scaling study ───────────────────────────────────────────────
    # For each scale, extract a contiguous block of A-lines from the center
    # and create a properly sized forward model + sub-problem.
    config_base = OmegaConf.load(
        os.path.join(PHANTOM_ROOT, 'configs', 'config.yaml'))
    config_base.forward_model_system.forward_model_type = 'ss'

    # ── Generate initial conditions at FULL size, once ──────────────────
    # Each (seed) has one full-size random μ array; per-scale subproblems
    # take a contiguous slice from the center. This guarantees that for
    # a fixed tissue pixel position, the initial μ is identical across
    # scales — so variations in recovery accuracy are attributable to
    # optimizer behavior, not to different random initializations per scale.
    # Tissue pixels are drawn uniform in [MU_MIN, 10.0] rather than
    # [0, 10.0] so every initial value is strictly feasible for bounded
    # optimizers (pycma's CMA-ES and scipy's L-BFGS-B / dual_annealing
    # all verify that x0 ∈ [MU_MIN, MU_MAX] at constructor time).
    INIT_SEED_BASE = 42
    INIT_MU_HIGH = 10.0
    mu_inits_full_np = []  # list of NUM_SEEDS float32 arrays of shape (H, W_full)
    tissue_mask_full_np = (mu_gt_np_full > MU_AIR * 10).astype(np.float32)
    for seed_idx in range(NUM_SEEDS):
        rng = np.random.default_rng(seed=INIT_SEED_BASE + seed_idx)
        mu_scr = rng.uniform(
            MU_MIN, INIT_MU_HIGH,
            size=mu_gt_np_full.shape).astype(np.float32)
        mu_init_full = (mu_scr * tissue_mask_full_np
                        + MU_AIR * (1 - tissue_mask_full_np))
        mu_inits_full_np.append(mu_init_full)
    print(f"\n  Generated {NUM_SEEDS} full-size initial μ arrays "
          f"(seeds {INIT_SEED_BASE}..{INIT_SEED_BASE + NUM_SEEDS - 1}, "
          f"uniform in [{MU_MIN}, {INIT_MU_HIGH}] on tissue); "
          f"each scale slices the same center block.")

    # Results: scale -> opt_name -> {time_ms, mape, timed_out}
    all_scale_results = {}

    for n_alines in scales_to_run:
        print(f"\n{'#'*70}")
        print(f"  Scale: {n_alines} A-lines")
        print(f"{'#'*70}")

        # Extract contiguous A-line block from center of 500-wide phantom
        center = mu_gt_np_full.shape[1] // 2
        a0 = max(0, center - n_alines // 2)
        a1 = a0 + n_alines
        if a1 > mu_gt_np_full.shape[1]:
            a1 = mu_gt_np_full.shape[1]
            a0 = a1 - n_alines

        mu_gt_sub = mu_gt_full[:, a0:a1].contiguous()
        alpha_gt_sub = alpha_gt_full[:, a0:a1].contiguous()
        target_sub = target_dB_full[:, a0:a1].contiguous()
        mu_gt_np_sub = mu_gt_np_full[:, a0:a1]

        depth = mu_gt_sub.shape[0]
        tissue_bool_sub = mu_gt_sub > MU_AIR * 10
        n_tissue = int(tissue_bool_sub.sum().item())
        n_total = mu_gt_sub.numel()
        print(f"  Shape: {depth} x {n_alines} = {n_total} pixels "
              f"({n_tissue} tissue)")

        # Create forward model for this scale
        config_sub = OmegaConf.create(OmegaConf.to_container(config_base))
        config_sub.forward_model_system.a_lines = n_alines
        fwd = create_forward_model(config_sub, device=device)
        fwd.gain_dB.fill_(gain_dB_val)
        fwd.eval()

        noise_mask_sub = build_noise_floor_mask(target_sub, NOISE_FLOOR_DB,
                                                device)

        # Slice initial conditions from the shared full-size arrays so
        # that every scale sees the same random μ at every tissue-pixel
        # position it contains. See the "Generate initial conditions at
        # FULL size" block above main()'s scale loop. Only build inits
        # for the seeds we will actually run (saves memory under SEED_INDEX).
        inits_sub = {}
        for seed_idx in seeds_to_run:
            mu_init_sliced = mu_inits_full_np[seed_idx][:, a0:a1].copy()
            inits_sub[seed_idx] = torch.tensor(mu_init_sliced,
                                               dtype=torch.float32,
                                               device=device)

        scale_results = {}

        for opt_name in OPTIMIZER_ORDER:
            opt_fn = OPTIMIZERS[opt_name]
            seed_times, seed_mapes, seed_fwd_evals = [], [], []
            seed_wall_sec, seed_peak_mem_mb, seed_ms_per_fwd = [], [], []

            for seed_idx in seeds_to_run:
                (mu_final, avg_time_ms, mse_history, fwd_evals,
                 total_wall_sec, peak_mem_mb) = opt_fn(
                    fwd, inits_sub[seed_idx], alpha_gt_sub, target_sub,
                    noise_mask_sub, device, NUM_ITERS, seed_idx=seed_idx)

                mape = compute_mape(mu_final, mu_gt_sub, tissue_bool_sub)
                seed_times.append(avg_time_ms)
                seed_mapes.append(mape)
                seed_fwd_evals.append(fwd_evals)
                seed_wall_sec.append(total_wall_sec)
                seed_peak_mem_mb.append(peak_mem_mb)
                seed_ms_per_fwd.append(
                    (total_wall_sec / max(fwd_evals, 1)) * 1000.0)
                del mu_final

            # When running a single seed (SEED_INDEX shard), avg/std fields
            # collapse to that single value / 0; the merger across seeds
            # rebuilds proper avg+std afterwards. Per-seed raw values are
            # kept under 'seed_<key>' for the merger to reconstruct stats.
            s = {
                'avg_time_ms': float(np.mean(seed_times)),
                'std_time_ms': float(np.std(seed_times)),
                'avg_mape': float(np.mean(seed_mapes)),
                'std_mape': float(np.std(seed_mapes)),
                'avg_fwd_evals': float(np.mean(seed_fwd_evals)),
                'std_fwd_evals': float(np.std(seed_fwd_evals)),
                'avg_wall_sec': float(np.mean(seed_wall_sec)),
                'std_wall_sec': float(np.std(seed_wall_sec)),
                'avg_ms_per_fwd': float(np.mean(seed_ms_per_fwd)),
                'std_ms_per_fwd': float(np.std(seed_ms_per_fwd)),
                'avg_peak_mem_mb': float(np.mean(seed_peak_mem_mb)),
                'std_peak_mem_mb': float(np.std(seed_peak_mem_mb)),
                'seed_time_ms':       seed_times,
                'seed_mape':          seed_mapes,
                'seed_fwd_evals':     seed_fwd_evals,
                'seed_wall_sec':      seed_wall_sec,
                'seed_ms_per_fwd':    seed_ms_per_fwd,
                'seed_peak_mem_mb':   seed_peak_mem_mb,
                'seeds_present':      list(seeds_to_run),
            }
            scale_results[opt_name] = s
            print(f"  {opt_name:25s}  "
                  f"wall={s['avg_wall_sec']:.2f}+-{s['std_wall_sec']:.2f} s  "
                  f"ms/fwd={s['avg_ms_per_fwd']:.3f}  "
                  f"MAPE={s['avg_mape']:.2f}+-{s['std_mape']:.2f}%  "
                  f"fwd_evals={s['avg_fwd_evals']:.0f}  "
                  f"peak={s['avg_peak_mem_mb']:.0f}MB")

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        all_scale_results[n_alines] = scale_results
        del fwd, noise_mask_sub, mu_gt_sub, alpha_gt_sub, target_sub

    # ── SCALE_INDEX shard: one task per scale; aggregate via afterok job ──
    if scale_shard:
        n_alines = scales_to_run[0]
        shard_path = os.path.join(OUT_DIR,
                                  f'results_scale{n_alines:04d}.json')
        with open(shard_path, 'w') as f:
            json.dump({
                str(n_alines): all_scale_results[n_alines],
                '_provenance': collect_provenance(),
            }, f, indent=2, default=str)
        print(f"\n  Shard saved to {shard_path}")
        print("  (Aggregate via MERGE_BASELINE=1 once all scale shards land.)")
        return

    # ── SEED_INDEX shard: one task per seed; aggregate via MERGE_PER_SEED=1 ─
    if seed_shard:
        shard_path = os.path.join(OUT_DIR, f'results_seed{single_seed}.json')
        # Strip per-scale 'seeds_present' lists down to the actual seed run
        # (already a singleton list by construction) so the merger sees clean
        # input. Convert int keys to strings for JSON.
        payload = {str(n): scale_results
                   for n, scale_results in all_scale_results.items()}
        payload['_seed'] = single_seed
        payload['_provenance'] = collect_provenance()
        with open(shard_path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n  Per-seed shard saved to {shard_path}")
        print("  (Aggregate via MERGE_PER_SEED=1 once all seed shards land.)")
        return

    # ── Save results JSON ────────────────────────────────────────────────
    results_path = os.path.join(OUT_DIR, 'results.json')
    payload = dict(all_scale_results)
    payload['_provenance'] = collect_provenance()
    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # ── Stage the bare-minimum subset for the Figure 1 notebook ─────────
    _stage_to_figure_paper_1(OUT_DIR)


def _stage_to_figure_paper_1(out_dir):
    """Copy the aggregated per-scale results_scale{N:04d}.json files (the only
    inputs the Figure 1 notebook reads) into figures/data/figureS1_optimizer_scaling/bscan/.
    Per-seed shards (results_scale{N}_seed{S}.json) stay under outputs/ as
    intermediate aggregation inputs."""
    import shutil, glob
    fig_paper_1 = os.path.join(PHANTOM_ROOT, 'figures', 'data', 'figureS1_optimizer_scaling/bscan')
    os.makedirs(fig_paper_1, exist_ok=True)
    n_staged = 0
    for src in sorted(glob.glob(os.path.join(out_dir, 'results_scale[0-9]*.json'))):
        # exclude per-seed shards
        if '_seed' in os.path.basename(src):
            continue
        shutil.copy2(src, os.path.join(fig_paper_1, os.path.basename(src)))
        n_staged += 1
    print(f"  [stage \u2192 figureS1_optimizer_scaling/bscan] {n_staged} aggregated JSON files in {fig_paper_1}")

    # Reload merged stats from staged JSONs so the LaTeX/plot helpers don't
    # depend on locals from main(). Skip gracefully if phantom data missing \u2014
    # the notebook only needs the staged JSONs.
    import glob as _glob
    merged = {}
    for p in sorted(_glob.glob(os.path.join(out_dir, 'results_scale[0-9]*.json'))):
        if '_seed' in os.path.basename(p):
            continue
        with open(p) as f:
            d = json.load(f)
        for k, v in d.items():
            if not k.startswith('_'):
                merged[int(k)] = v
    if not merged:
        print("  [stage] no merged scale data; skipping LaTeX + plot")
        return
    try:
        mu_gt_np_full = np.load(os.path.join(PHANTOM_DATA_DIR, 'mu_gt_mm.npy'))
    except FileNotFoundError:
        print(f"  [stage] phantom data missing at {PHANTOM_DATA_DIR}; "
              "skipping LaTeX + plot (notebook still works from staged JSONs)")
        return
    scales_present = sorted(merged.keys())

    # ── LaTeX table (rows = scale, cols = optimizer) ────────────────────
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Optimizer scaling comparison on synthetic OCT phantom "
        r"recovery. Each metric is averaged over " + str(NUM_SEEDS) +
        r" random initializations with " + f"{NUM_ITERS:,}" +
        r" iterations. Gradient-based methods (Adam, L-BFGS-B) maintain "
        r"sub-1\% MAPE across all problem sizes, while gradient-free "
        r"methods (Random Search, CMA-ES) degrade rapidly with "
        r"increasing dimensionality.}",
        r"\label{tab:optimizer_baseline}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{r r || c c | c c}",
        r"\toprule",
        r" & & \multicolumn{2}{c|}{Gradient-free} "
        r"& \multicolumn{2}{c}{Gradient-based} \\",
        r"A-lines & Tissue px "
        r"& Random Search & CMA-ES "
        r"& Adam & L-BFGS-B \\",
        r"\midrule",
    ]

    for n_al in scales_present:
        sr = merged[n_al]
        center = mu_gt_np_full.shape[1] // 2
        a0 = max(0, center - n_al // 2)
        a1 = a0 + n_al
        n_tis = int((mu_gt_np_full[:, a0:a1] > MU_AIR * 10).sum())

        row_cells = [f"{n_al}", f"{n_tis:,}"]
        for opt_name in OPTIMIZER_ORDER:
            s = sr.get(opt_name, {})
            row_cells.append(
                f"${s.get('avg_mape', float('nan')):.2f} \\pm "
                f"{s.get('std_mape', float('nan')):.2f}$")

        latex_lines.append(" & ".join(row_cells) + r" \\")

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    latex_str = "\n".join(latex_lines)
    latex_path = os.path.join(out_dir, 'summary_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    print(f"  LaTeX table saved to {latex_path}")
    print(f"\n{latex_str}")

    # ── Scaling plot: MAPE vs A-lines for each optimizer ─────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'v', 'P']

    for i, opt_name in enumerate(OPTIMIZER_ORDER):
        scales, mapes, times = [], [], []
        for n_al in scales_present:
            if opt_name not in merged[n_al]:
                continue
            s = merged[n_al][opt_name]
            scales.append(n_al)
            mapes.append(s['avg_mape'])
            times.append(s['avg_time_ms'])

        if scales:
            ax1.plot(scales, mapes, color=colors[i], marker=markers[i],
                     linewidth=2, markersize=7, label=opt_name)
            ax2.plot(scales, times, color=colors[i], marker=markers[i],
                     linewidth=2, markersize=7, label=opt_name)

    ax1.set_xlabel('Number of A-lines', fontsize=12)
    ax1.set_ylabel('MAPE (%)', fontsize=12)
    ax1.set_title('Recovery Accuracy vs Problem Size', fontsize=13)
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Number of A-lines', fontsize=12)
    ax2.set_ylabel('Avg Iteration Time (ms)', fontsize=12)
    ax2.set_title('Iteration Time vs Problem Size', fontsize=13)
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'scaling_plot.png')
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Scaling plot saved to {fig_path}")

    print("\nDone!")


# ═══════════════════════════════════════════════════════════════════════════
# Convergence study: tracked optimizers with per-iteration (MAPE, R²)
# ═══════════════════════════════════════════════════════════════════════════
#
# Additive block — enabled only when GENERATE_CONV_PLOT=1. Does not touch
# any of the existing optimizer functions or main(). Each *_tracked variant
# mirrors its base optimizer exactly, adding a MAPE+R² snapshot at every
# LOG_INTERVAL iterations plus iter 0 and iter NUM_ITERS (after best-model
# restoration, where applicable).

CONV_OUT_DIR = os.path.join(PHANTOM_ROOT, 'outputs', f'optimizer_scaling{_OUT_SUFFIX}')


def compute_mape_r2(mu_recovered, mu_gt, tissue_bool):
    """Return (MAPE %, concordance R²) on the tissue pixel subset."""
    with torch.no_grad():
        mu_rec = mu_recovered[tissue_bool].cpu().numpy().astype(np.float64)
        mu_ref = mu_gt[tissue_bool].cpu().numpy().astype(np.float64)
    abs_err = np.abs(mu_rec - mu_ref)
    mape = float((abs_err / (mu_ref + 1e-12)).mean() * 100)
    if mu_rec.std() < 1e-12 or mu_ref.std() < 1e-12 or mu_rec.size < 2:
        r2 = 0.0
    else:
        corr = np.corrcoef(mu_rec, mu_ref)[0, 1]
        r2 = float(corr * corr) if np.isfinite(corr) else 0.0
    return mape, r2


def _append_snapshot(history, it, mu, mu_gt, tissue_bool):
    mape, r2 = compute_mape_r2(mu, mu_gt, tissue_bool)
    history.append((int(it), mape, r2))


def run_adam_tracked(forward_model, mu_init, alpha, target, noise_mask, device,
                     num_iters, mu_gt, tissue_bool):
    """Tracked Adam — mirrors run_adam + per-iter MAPE/R² history."""
    mu = torch.nn.Parameter(mu_init.clone())
    alpha_param = torch.nn.Parameter(alpha.clone(), requires_grad=False)
    optimizer = torch.optim.Adam([mu, alpha_param], lr=LR)

    with torch.no_grad():
        mu.data *= noise_mask

    best_mse = float('inf')
    best_mu = mu.detach().clone()

    # CUDA kernel warmup: one untimed forward+backward, no optimizer step.
    pred = forward_model(mu, alpha_param, noise_mask=noise_mask)
    loss = compute_loss(pred, target, noise_mask, LOSS_SCALE)
    loss.backward()
    optimizer.zero_grad(set_to_none=False)

    history = []
    _append_snapshot(history, 0, mu.detach(), mu_gt, tissue_bool)
    fwd_evals = 0

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=False)
        pred = forward_model(mu, alpha_param, noise_mask=noise_mask)
        fwd_evals += 1
        loss = compute_loss(pred, target, noise_mask, LOSS_SCALE)

        mse_val = loss.item() / LOSS_SCALE
        if mse_val < best_mse:
            best_mse = mse_val
            best_mu.copy_(mu.detach())

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mu.data.clamp_(MU_MIN, MU_MAX)
            mu.data.mul_(noise_mask)

        if (it + 1) % LOG_INTERVAL == 0 and (it + 1) < num_iters:
            _append_snapshot(history, it + 1, mu.detach(), mu_gt, tissue_bool)

    mu.data.copy_(best_mu)
    _append_snapshot(history, num_iters, mu.detach(), mu_gt, tissue_bool)
    return mu.detach(), history, fwd_evals


def run_lbfgs_tracked(forward_model, mu_init, alpha, target, noise_mask, device,
                      num_iters, mu_gt, tissue_bool):
    """Tracked L-BFGS-B (Byrd et al. 1995) via scipy.optimize.minimize.
    Same algorithm as run_lbfgs with per-iter MAPE/R² snapshots for the
    convergence figure via scipy's callback hook.
    """
    tissue_idx = torch.nonzero(noise_mask.flatten() > 0.5,
                               as_tuple=False).squeeze(1)
    n_tissue = int(tissue_idx.numel())

    mu_init_masked = mu_init.clone()
    mu_init_masked.mul_(noise_mask)
    x0_np = mu_init_masked.view(-1)[tissue_idx].detach().cpu().numpy().astype(np.float64)

    mu_work = torch.nn.Parameter(mu_init_masked.clone(), requires_grad=True)

    call_count = [0]
    outer_iter = [0]
    best = {'loss': float('inf'), 'x': x0_np.copy()}
    best_mu = mu_init_masked.clone()

    history = []
    _append_snapshot(history, 0, best_mu, mu_gt, tissue_bool)

    def fun_and_grad(x_np):
        x_tensor = torch.from_numpy(x_np.astype(np.float32)).to(device)
        with torch.no_grad():
            mu_work.data.view(-1)[tissue_idx] = x_tensor
        if mu_work.grad is not None:
            mu_work.grad.zero_()
        pred = forward_model(mu_work, alpha, noise_mask=noise_mask)
        loss = compute_loss(pred, target, noise_mask, LOSS_SCALE)
        loss.backward()
        loss_val = float(loss.item())
        grad_tissue = (mu_work.grad.view(-1)[tissue_idx]
                       .detach().cpu().numpy().astype(np.float64))
        if loss_val < best['loss']:
            best['loss'] = loss_val
            best['x'] = x_np.copy()
            with torch.no_grad():
                best_mu.view(-1)[tissue_idx] = x_tensor
        call_count[0] += 1
        return loss_val, grad_tissue

    def callback(xk):
        outer_iter[0] += 1
        if (outer_iter[0] % LOG_INTERVAL == 0
                and outer_iter[0] < num_iters):
            _append_snapshot(history, outer_iter[0], best_mu, mu_gt, tissue_bool)

    bounds = [(MU_MIN, MU_MAX)] * n_tissue

    scipy_minimize(
        fun_and_grad,
        x0=x0_np,
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        callback=callback,
        options={
            'maxiter': num_iters,
            'maxfun': num_iters * 20,
            'ftol': 0.0,
            'gtol': 0.0,
            'maxcor': 10,
        },
    )

    _append_snapshot(history, num_iters, best_mu, mu_gt, tissue_bool)
    return best_mu, history, call_count[0]


def run_random_search_tracked(forward_model, mu_init, alpha, target, noise_mask,
                               device, num_iters, mu_gt, tissue_bool):
    """Tracked Solis & Wets 1981 Basic Algorithm + Algorithm 1 sampling
    rule. Same algorithm as run_random_search with per-iter MAPE/R²
    snapshots for the convergence figure.
    """
    S_EX = 5
    F_CT = 3
    EX_FAC = 2.0
    CT_FAC = 0.5
    RHO_INIT = 1.0
    RHO_LB = 0.0

    mu = mu_init.clone()
    mu.mul_(noise_mask)

    rho = RHO_INIT
    bias = torch.zeros_like(mu)
    num_s = 0
    num_f = 0

    current_loss = eval_loss_nograph(forward_model, mu, alpha, target, noise_mask)
    best_loss = current_loss
    best_mu = mu.clone()

    history = []
    _append_snapshot(history, 0, best_mu, mu_gt, tissue_bool)
    fwd_evals = 0

    for it in range(num_iters):
        # ρ update per S&W 1981 Algorithm 1 Step 1 (p. 25). Counters are
        # reset only in Step 2, not at Step 1.
        if num_s >= S_EX:
            rho = rho * EX_FAC
        elif num_f >= F_CT:
            rho = rho * CT_FAC
        if rho <= RHO_LB:
            break

        eps = torch.randn_like(mu) * noise_mask
        xi = mu + bias + rho * eps
        xi_proj = xi.clamp(MU_MIN, MU_MAX) * noise_mask
        loss_xi = eval_loss_nograph(forward_model, xi_proj, alpha, target,
                                    noise_mask)
        fwd_evals += 1

        if loss_xi < current_loss:
            bias = 0.4 * (xi - mu) + 0.2 * bias
            mu = xi_proj
            current_loss = loss_xi
            num_s += 1
            num_f = 0
        else:
            rev = 2.0 * mu - xi
            rev_proj = rev.clamp(MU_MIN, MU_MAX) * noise_mask
            loss_rev = eval_loss_nograph(forward_model, rev_proj, alpha,
                                         target, noise_mask)
            fwd_evals += 1
            if loss_rev < current_loss:
                bias = bias - 0.4 * (xi - mu)
                mu = rev_proj
                current_loss = loss_rev
                num_s += 1
                num_f = 0
            else:
                bias = 0.5 * bias
                num_s = 0
                num_f += 1

        if current_loss < best_loss:
            best_loss = current_loss
            best_mu = mu.clone()

        if (it + 1) % LOG_INTERVAL == 0 and (it + 1) < num_iters:
            _append_snapshot(history, it + 1, best_mu, mu_gt, tissue_bool)

    _append_snapshot(history, num_iters, best_mu, mu_gt, tissue_bool)
    return best_mu, history, fwd_evals



def run_cma_es_tracked(forward_model, mu_init, alpha, target,
                        noise_mask, device, num_iters, mu_gt,
                        tissue_bool):
    """Tracked Hansen & Ostermeier 2001 CMA-ES via pycma. Same algorithm
    as run_cma_es with per-iter MAPE/R² snapshots for the convergence
    figure.
    """
    tissue_idx = torch.nonzero(noise_mask.flatten() > 0.5,
                               as_tuple=False).squeeze(1)
    n_tissue = int(tissue_idx.numel())

    mu_init_masked = mu_init.clone()
    mu_init_masked.mul_(noise_mask)
    x0 = mu_init_masked.view(-1)[tissue_idx].detach().cpu().numpy().astype(np.float64)
    sigma0 = 1.0

    mu_work = mu_init_masked.clone()

    call_count = [0]
    best = {'loss': float('inf'),
            'x': x0.copy()}
    best_mu = mu_init_masked.clone()

    history = []
    _append_snapshot(history, 0, best_mu, mu_gt, tissue_bool)

    def objective(x_np):
        x_arr = np.asarray(x_np, dtype=np.float32)
        x_tensor = torch.from_numpy(x_arr).to(device)
        with torch.no_grad():
            mu_work.view(-1)[tissue_idx] = x_tensor
        loss = eval_loss_nograph(forward_model, mu_work, alpha, target,
                                 noise_mask)
        if loss < best['loss']:
            best['loss'] = loss
            best['x'] = np.asarray(x_np, dtype=np.float64).copy()
            with torch.no_grad():
                best_mu.view(-1)[tissue_idx] = x_tensor
        call_count[0] += 1
        if call_count[0] % LOG_INTERVAL == 0 and call_count[0] < num_iters:
            _append_snapshot(history, call_count[0], best_mu, mu_gt, tissue_bool)
        return loss

    opts = {
        'bounds': [[MU_MIN] * n_tissue, [MU_MAX] * n_tissue],
        'maxfevals': num_iters,
        'seed': 1,
        'verbose': -9,
        'tolfun': 0.0,
        'tolx': 0.0,
        'CMA_active': False,
        'updatecovwait': max(1, int(math.ceil(math.sqrt(n_tissue)))),  # H&O 2001 §5.2 √n option
    }

    try:
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
        while not es.stop() and call_count[0] < num_iters:
            solutions = es.ask()
            fitnesses = [objective(x) for x in solutions]
            if call_count[0] >= num_iters:
                break
            es.tell(solutions, fitnesses)
    except (MemoryError, np.linalg.LinAlgError) as e:
        print(f"  [CMA-ES tracked] terminated early (n={n_tissue}): {type(e).__name__}: {e}")

    _append_snapshot(history, num_iters, best_mu, mu_gt, tissue_bool)
    return best_mu, history, call_count[0]


OPTIMIZERS_TRACKED = {
    'Adam': run_adam_tracked,
    'L-BFGS': run_lbfgs_tracked,
    'Random Search': run_random_search_tracked,
    'CMA-ES': run_cma_es_tracked,
}


def plot_convergence_figure(history_json_path, fig_path):
    """Convergence figure mimicking panel (t) of forward_inverse_verification_src.pdf.
    Style copied from notebooks/figureS1_optimizer_scaling/bscan.ipynb: sans-serif Helvetica,
    colored axis labels (MAPE red, R² blue), symlog x with linthresh=100,
    thin spines (0.9), LABEL_FS_BIG=15, TICK_FS_BIG=15."""
    with open(history_json_path) as f:
        hist = json.load(f)

    from matplotlib.lines import Line2D

    # ── apply_paper_style() from notebooks/figureS1_optimizer_scaling/bscan.ipynb ────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 120,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
    })
    LABEL_FS = 13
    TICK_FS = 13
    LEGEND_FS = 12
    LABEL_FS_BIG = LABEL_FS + 2   # 15
    TICK_FS_BIG = TICK_FS + 2     # 15

    grad_based = ['Adam', 'L-BFGS']
    grad_free = ['Random Search', 'CMA-ES']
    scales_to_plot = [25]
    all_methods = grad_based + grad_free

    method_colors = {
        'Adam':                '#1f77b4',
        'L-BFGS':              '#ff7f0e',
        'Random Search':       '#2ca02c',
        'CMA-ES':              '#9467bd',
    }

    # Panel (t) exact axis colors
    color_mape = '#E24A33'
    color_r2 = '#348ABD'

    fig, ax_left = plt.subplots(1, 1, figsize=(8, 5.5))
    ax_right = ax_left.twinx()

    for opt in all_methods:
        color = method_colors[opt]
        for s in scales_to_plot:
            key_s = str(s)
            if key_s not in hist or opt not in hist[key_s]:
                continue
            h = hist[key_s][opt]
            iters = np.array(h['iters'], dtype=float)
            mape = np.array(h['mape_mean'], dtype=float)
            r2 = np.array(h['r2_mean'], dtype=float)
            ax_left.plot(iters, mape, color=color,
                         linestyle='-', linewidth=2.0)
            ax_right.plot(iters, r2, color=color,
                          linestyle='--', linewidth=2.0)

    # ── style_ax() from notebook: spines, ticks ────────────────────────
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['left'].set_linewidth(0.9)
    ax_left.spines['bottom'].set_linewidth(0.9)
    ax_left.tick_params(axis='both', labelsize=TICK_FS_BIG,
                        width=0.8, length=3.5)

    # Panel (t): symlog x with linthresh=100, left y linear from 0
    ax_left.set_xscale('symlog', linthresh=100)
    ax_left.set_xlim(0, NUM_ITERS)
    ax_left.set_ylim(0, 100)
    ax_left.set_xlabel('Iteration', fontsize=LABEL_FS_BIG)
    ax_left.set_ylabel('MAPE [%]', fontsize=LABEL_FS_BIG, color=color_mape)
    ax_left.tick_params(axis='y', labelcolor=color_mape,
                        labelsize=TICK_FS_BIG)
    ax_left.grid(False)

    # Right axis (R²): restore right spine per notebook
    ax_right.set_ylim(0, 1.05)
    ax_right.set_ylabel(r'$R^2$', fontsize=LABEL_FS_BIG, color=color_r2)
    ax_right.tick_params(axis='y', labelcolor=color_r2,
                        labelsize=TICK_FS_BIG)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(True)
    ax_right.spines['right'].set_linewidth(0.9)
    ax_right.grid(False)

    # Method legend only — metric encoded by colored axis labels + line style.
    # Frame matches notebook concordance-panel annotation style (rounded white
    # box with #bbbbbb edge, alpha 0.9).
    method_handles = [
        Line2D([0], [0], color=method_colors[m], linestyle='-', linewidth=2.0,
               label=m)
        for m in all_methods
    ]
    leg = ax_left.legend(
        handles=method_handles, loc='upper left',
        fontsize=LEGEND_FS, fancybox=True, framealpha=0.9,
        facecolor='white', edgecolor='#bbbbbb',
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main_convergence_study():
    # If SCALE_INDEX is set, run only that one scale (sharded GPU mode).
    shard_mode = SCALE_INDEX != ''
    if shard_mode:
        try:
            shard_idx = int(SCALE_INDEX)
            scales_to_run = [ALINE_SCALES[shard_idx]]
        except (ValueError, IndexError):
            raise SystemExit(f"Invalid SCALE_INDEX={SCALE_INDEX!r}; "
                             f"must be integer in [0, {len(ALINE_SCALES)-1}]")
    else:
        scales_to_run = list(ALINE_SCALES)

    print("=" * 70)
    print("  Convergence-plot study (tracked MAPE + R² per iter)")
    if shard_mode:
        print(f"  SHARD mode: SCALE_INDEX={SCALE_INDEX} -> "
              f"{scales_to_run[0]} A-lines only")
    print(f"  {len(OPTIMIZER_ORDER)} optimizers x {len(scales_to_run)} scales "
          f"x {NUM_SEEDS} seeds x {NUM_ITERS:,} iters")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CONV_OUT_DIR, exist_ok=True)

    print(f"\nLoading phantom data from:\n  {PHANTOM_DATA_DIR}")
    mu_gt_full, alpha_gt_full, target_dB_full, mu_gt_np_full, gain_dB_val = \
        load_phantom_data(device)
    print(f"  Full phantom shape: {mu_gt_np_full.shape}")

    config_base = OmegaConf.load(
        os.path.join(PHANTOM_ROOT, 'configs', 'config.yaml'))
    config_base.forward_model_system.forward_model_type = 'ss'

    # Generate full-size initial μ arrays once; slice per scale (see
    # matching comment in main() — same shared-seed design). Sampled in
    # [MU_MIN, 10.0] so every init is strictly inside bounds for CMA-ES
    # and L-BFGS-B.
    INIT_SEED_BASE = 42
    INIT_MU_HIGH = 10.0
    mu_inits_full_np = []
    tissue_mask_full_np = (mu_gt_np_full > MU_AIR * 10).astype(np.float32)
    for seed_idx in range(NUM_SEEDS):
        rng = np.random.default_rng(seed=INIT_SEED_BASE + seed_idx)
        mu_scr = rng.uniform(
            MU_MIN, INIT_MU_HIGH,
            size=mu_gt_np_full.shape).astype(np.float32)
        mu_init_full = (mu_scr * tissue_mask_full_np
                        + MU_AIR * (1 - tissue_mask_full_np))
        mu_inits_full_np.append(mu_init_full)

    all_histories = {}
    t_study_start = time.time()

    for n_alines in scales_to_run:
        print(f"\n{'#'*70}")
        print(f"  Scale: {n_alines} A-lines")
        print(f"{'#'*70}")

        center = mu_gt_np_full.shape[1] // 2
        a0 = max(0, center - n_alines // 2)
        a1 = a0 + n_alines
        if a1 > mu_gt_np_full.shape[1]:
            a1 = mu_gt_np_full.shape[1]
            a0 = a1 - n_alines

        mu_gt_sub = mu_gt_full[:, a0:a1].contiguous()
        alpha_gt_sub = alpha_gt_full[:, a0:a1].contiguous()
        target_sub = target_dB_full[:, a0:a1].contiguous()
        mu_gt_np_sub = mu_gt_np_full[:, a0:a1]

        tissue_bool_sub = mu_gt_sub > MU_AIR * 10

        config_sub = OmegaConf.create(OmegaConf.to_container(config_base))
        config_sub.forward_model_system.a_lines = n_alines
        fwd = create_forward_model(config_sub, device=device)
        fwd.gain_dB.fill_(gain_dB_val)
        fwd.eval()

        noise_mask_sub = build_noise_floor_mask(target_sub, NOISE_FLOOR_DB,
                                                device)

        inits_sub = []
        for seed_idx in range(NUM_SEEDS):
            mu_init_sliced = mu_inits_full_np[seed_idx][:, a0:a1].copy()
            inits_sub.append(torch.tensor(mu_init_sliced,
                                          dtype=torch.float32,
                                          device=device))

        scale_hist = {}

        for opt_name in OPTIMIZER_ORDER:
            opt_fn = OPTIMIZERS_TRACKED[opt_name]
            seed_histories = []
            t_opt_start = time.time()
            for seed_idx in range(NUM_SEEDS):
                _, history, _ = opt_fn(
                    fwd, inits_sub[seed_idx], alpha_gt_sub, target_sub,
                    noise_mask_sub, device, NUM_ITERS,
                    mu_gt_sub, tissue_bool_sub)
                seed_histories.append(history)

            # Align on the common iter grid (all seeds use identical logging)
            iters_ref = [pt[0] for pt in seed_histories[0]]
            n_pts = len(iters_ref)
            mape_arr = np.array(
                [[pt[1] for pt in h[:n_pts]] for h in seed_histories])
            r2_arr = np.array(
                [[pt[2] for pt in h[:n_pts]] for h in seed_histories])

            scale_hist[opt_name] = {
                'iters': iters_ref,
                'mape_mean': mape_arr.mean(axis=0).tolist(),
                'mape_std': mape_arr.std(axis=0).tolist(),
                'r2_mean': r2_arr.mean(axis=0).tolist(),
                'r2_std': r2_arr.std(axis=0).tolist(),
            }
            print(f"  {opt_name:25s}  "
                  f"final MAPE={mape_arr[:, -1].mean():.2f}%  "
                  f"final R²={r2_arr[:, -1].mean():.4f}  "
                  f"({time.time() - t_opt_start:.1f}s)")

            if device.type == 'cuda':
                torch.cuda.empty_cache()

        all_histories[str(n_alines)] = scale_hist
        del fwd, noise_mask_sub, mu_gt_sub, alpha_gt_sub, target_sub

        if shard_mode:
            shard_path = os.path.join(
                CONV_OUT_DIR, f'convergence_history_scale{n_alines:04d}.json')
            with open(shard_path, 'w') as f:
                json.dump({str(n_alines): scale_hist}, f, indent=2)
            print(f"  Shard saved to {shard_path}")
        else:
            hist_path = os.path.join(CONV_OUT_DIR, 'convergence_history.json')
            with open(hist_path, 'w') as f:
                json.dump(all_histories, f, indent=2)

    total_study_min = (time.time() - t_study_start) / 60
    print(f"\n  Study wall time: {total_study_min:.1f} min")

    if shard_mode:
        print("\nShard complete. Run MERGE_CONV=1 "
              "python test/optimizer_scaling.py to aggregate + plot.")
        return

    hist_path = os.path.join(CONV_OUT_DIR, 'convergence_history.json')
    print(f"  History saved to {hist_path}")

    fig_path = os.path.join(CONV_OUT_DIR, 'convergence_vs_iter.pdf')
    plot_convergence_figure(hist_path, fig_path)
    print(f"  Figure saved to {fig_path}")

    print("\nDone!")


def merge_baseline_shards():
    """Aggregate per-scale scaling-study shards into results.json and
    regenerate the LaTeX table + scaling plot. Invoked via
    MERGE_BASELINE=1 env var after all array-job scales have completed.
    """
    import glob as _glob
    os.makedirs(OUT_DIR, exist_ok=True)
    # Only aggregated per-scale shards (results_scale{N:04d}.json);
    # skip per-seed shards written by optimizer_scaling.py with NUM_ITERS=20000 and OUT_SUFFIX=_20k for the per-seed sweep
    # (they contain single-seed values and different field names).
    shard_paths = sorted(
        p for p in _glob.glob(os.path.join(OUT_DIR, 'results_scale*.json'))
        if '_seed' not in os.path.basename(p))
    if not shard_paths:
        raise SystemExit(f"No shard files found in {OUT_DIR}")

    merged = {}
    shard_provenance = {}
    for p in shard_paths:
        with open(p) as f:
            shard = json.load(f)
        # Separate provenance from scale entries; keep the first shard's
        # provenance as a representative (all shards run on the same image
        # but different array tasks may land on different GPUs).
        prov = shard.pop('_provenance', None)
        if prov is not None and not shard_provenance:
            shard_provenance = prov
        # Drop any metadata keys that start with an underscore — the
        # per-seed aggregator may write _seeds, future additions may use
        # _seeds_present, etc. Only integer scale keys survive.
        shard = {k: v for k, v in shard.items() if not k.startswith('_')}
        merged.update(shard)
        print(f"  merged {os.path.basename(p)}")
    # Convert string-keyed JSON back to int-keyed dict (matches main())
    all_scale_results = {int(k): v for k, v in merged.items()}

    results_path = os.path.join(OUT_DIR, 'results.json')
    payload = {k: v for k, v in all_scale_results.items()}
    payload['_provenance_merge'] = collect_provenance()
    if shard_provenance:
        payload['_provenance_first_shard'] = shard_provenance
    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print(f"  Scales in merged file: {sorted(all_scale_results.keys())}")

    # Reload full phantom data purely for tissue-pixel count in LaTeX rows
    # (no GPU work needed — use CPU)
    mu_gt_np_full = np.load(os.path.join(PHANTOM_DATA_DIR, 'mu_gt_mm.npy'))

    # ── LaTeX table ─────────────────────────────────────────────────────
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Optimizer scaling comparison on synthetic OCT phantom "
        r"recovery. Each metric is averaged over " + str(NUM_SEEDS) +
        r" random initializations with " + f"{NUM_ITERS:,}" +
        r" iterations.}",
        r"\label{tab:optimizer_baseline}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{r r || c c | c c}",
        r"\toprule",
        r" & & \multicolumn{2}{c|}{Gradient-free} "
        r"& \multicolumn{2}{c}{Gradient-based} \\",
        r"A-lines & Tissue px & "
        r"Rand.\ Search & CMA-ES "
        r"& Adam & L-BFGS-B \\",
        r"\midrule",
    ]
    for n_al in sorted(all_scale_results.keys()):
        center = mu_gt_np_full.shape[1] // 2
        a0 = max(0, center - n_al // 2)
        a1 = a0 + n_al
        n_tis = int((mu_gt_np_full[:, a0:a1] > MU_AIR * 10).sum())
        row_cells = [f"{n_al}", f"{n_tis:,}"]
        sr = all_scale_results[n_al]
        for opt_name in OPTIMIZER_ORDER:
            s = sr[opt_name]
            row_cells.append(
                f"${s['avg_mape']:.2f} \\pm {s['std_mape']:.2f}$")
        latex_lines.append(" & ".join(row_cells) + r" \\")
    latex_lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    latex_str = "\n".join(latex_lines)
    latex_path = os.path.join(OUT_DIR, 'summary_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    print(f"  LaTeX table saved to {latex_path}")

    # ── Scaling plot ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'v', 'P']
    scales_sorted = sorted(all_scale_results.keys())
    for i, opt_name in enumerate(OPTIMIZER_ORDER):
        mapes = [all_scale_results[n][opt_name]['avg_mape'] for n in scales_sorted]
        times = [all_scale_results[n][opt_name]['avg_time_ms'] for n in scales_sorted]
        ax1.plot(scales_sorted, mapes, color=colors[i], marker=markers[i],
                 linewidth=2, markersize=7, label=opt_name)
        ax2.plot(scales_sorted, times, color=colors[i], marker=markers[i],
                 linewidth=2, markersize=7, label=opt_name)
    ax1.set_xlabel('Number of A-lines', fontsize=12)
    ax1.set_ylabel('MAPE (%)', fontsize=12)
    ax1.set_title('Recovery Accuracy vs Problem Size', fontsize=13)
    ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.grid(alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax2.set_xlabel('Number of A-lines', fontsize=12)
    ax2.set_ylabel('Time per iter (ms)', fontsize=12)
    ax2.set_title('Per-Iteration Cost vs Problem Size', fontsize=13)
    ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.grid(alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'scaling_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Scaling plot saved to {plot_path}")
    print("\nDone!")


def merge_per_seed_shards():
    """Aggregate per-seed shards into per-scale results_scale{N:04d}.json.

    Reads every results_seed{S}.json under OUT_DIR and rebuilds avg/std
    statistics across seeds for each (scale, optimizer). Writes one
    aggregated file per scale, then stages figureS1_optimizer_scaling/bscan.
    """
    import glob as _glob
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_paths = sorted(_glob.glob(os.path.join(OUT_DIR, 'results_seed*.json')))
    if not seed_paths:
        raise SystemExit(f"No per-seed shards found in {OUT_DIR}")

    # Index per-seed shards: shards[seed_idx][scale_int][optimizer] = stats dict
    shards = {}
    for p in seed_paths:
        with open(p) as f:
            data = json.load(f)
        s_idx = data.get('_seed')
        if s_idx is None:
            print(f"  skipping {os.path.basename(p)} (no _seed key)")
            continue
        clean = {int(k): v for k, v in data.items() if not k.startswith('_')}
        shards[int(s_idx)] = clean
        print(f"  loaded results_seed{s_idx}.json ({len(clean)} scales)")

    if not shards:
        raise SystemExit("No usable per-seed shards.")

    seeds_sorted = sorted(shards.keys())
    scales_sorted = sorted({n for s in shards.values() for n in s.keys()})
    optimizers = list(OPTIMIZER_ORDER)
    print(f"\n  Aggregating across seeds {seeds_sorted}, "
          f"scales {scales_sorted}, optimizers {optimizers}")

    # Metric keys: per-seed list field name -> aggregated avg/std field names.
    seed_field_to_avg_std = {
        'seed_time_ms':     ('avg_time_ms',     'std_time_ms'),
        'seed_mape':        ('avg_mape',        'std_mape'),
        'seed_fwd_evals':   ('avg_fwd_evals',   'std_fwd_evals'),
        'seed_wall_sec':    ('avg_wall_sec',    'std_wall_sec'),
        'seed_ms_per_fwd':  ('avg_ms_per_fwd',  'std_ms_per_fwd'),
        'seed_peak_mem_mb': ('avg_peak_mem_mb', 'std_peak_mem_mb'),
    }

    n_staged = 0
    for n_alines in scales_sorted:
        per_opt = {}
        for opt_name in optimizers:
            # Gather raw per-seed values across all seed shards
            collected = {k: [] for k in seed_field_to_avg_std}
            for s_idx in seeds_sorted:
                if n_alines not in shards[s_idx]:
                    continue
                if opt_name not in shards[s_idx][n_alines]:
                    continue
                src = shards[s_idx][n_alines][opt_name]
                for k in collected:
                    if k in src and src[k]:
                        collected[k].extend(src[k])

            agg = {}
            for k, (avg_key, std_key) in seed_field_to_avg_std.items():
                vals = np.array(collected[k], dtype=float) if collected[k] else np.array([np.nan])
                agg[avg_key] = float(np.mean(vals))
                agg[std_key] = float(np.std(vals))
                agg[k] = collected[k]
            agg['seeds_present'] = seeds_sorted
            per_opt[opt_name] = agg

        out_path = os.path.join(OUT_DIR, f'results_scale{n_alines:04d}.json')
        payload = {str(n_alines): per_opt, '_provenance_merge': collect_provenance()}
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        n_staged += 1
        print(f"  wrote {os.path.basename(out_path)}")

    print(f"\n  Wrote {n_staged} aggregated per-scale files.")
    _stage_to_figure_paper_1(OUT_DIR)
    print("Done!")


def merge_conv_shards():
    """Aggregate per-scale shard JSONs into convergence_history.json and
    produce the convergence figure."""
    import glob as _glob
    os.makedirs(CONV_OUT_DIR, exist_ok=True)
    shard_paths = sorted(
        _glob.glob(os.path.join(CONV_OUT_DIR,
                                'convergence_history_scale*.json')))
    if not shard_paths:
        raise SystemExit(f"No shard files found in {CONV_OUT_DIR}")

    merged = {}
    for p in shard_paths:
        with open(p) as f:
            merged.update(json.load(f))
        print(f"  merged {os.path.basename(p)}")

    hist_path = os.path.join(CONV_OUT_DIR, 'convergence_history.json')
    with open(hist_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"\n  History saved to {hist_path}")
    print(f"  Scales in merged file: "
          f"{sorted(int(k) for k in merged.keys())}")

    fig_path = os.path.join(CONV_OUT_DIR, 'convergence_vs_iter.pdf')
    plot_convergence_figure(hist_path, fig_path)
    print(f"  Figure saved to {fig_path}")
    print(f"  PNG saved to {fig_path.replace('.pdf', '.png')}")
    print("\nDone!")


if __name__ == '__main__':
    if MERGE_PER_SEED:
        merge_per_seed_shards()
    elif MERGE_BASELINE:
        merge_baseline_shards()
    elif MERGE_CONV:
        merge_conv_shards()
    elif GENERATE_CONV_PLOT:
        main_convergence_study()
    else:
        main()
