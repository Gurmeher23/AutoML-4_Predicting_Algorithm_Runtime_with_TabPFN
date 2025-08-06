#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# 1) Add DistNet/ and src/ to PYTHONPATH
repo_root = os.getcwd()   # run this from inside DistNet/
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "src"))
from helper import load_data, data_source_release

# 2) Locate fig4_results_true one level up from DistNet/
base_dir = os.path.abspath(os.path.join(repo_root, os.pardir, "fig4_results_true"))
if not os.path.isdir(base_dir):
    raise FileNotFoundError(f"Can't find fig4_results_true at {base_dir}")

# 3) Discover scenarios
scenarios = sorted(
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
)

model_name   = "lognormal_nn"
task         = "floc"
seed         = 1
sample_sizes = [2, 4, 8, 16, 32, 64, 100]
n_folds      = 10

# 4) Load true runtimes for each scenario
sc_dict  = data_source_release.get_sc_dict()
true_rts = {}
for scen in scenarios:
    runtimes, _, _ = load_data.get_data(
        scenario=scen,
        data_dir=data_source_release.get_data_dir(),
        sc_dict=sc_dict,
        retrieve=sc_dict[scen]["use"]
    )
    # shape: (n_instances, n_obs)
    true_rts[scen] = np.array(runtimes)

kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

# 5) Compute mean log-likelihood per observation on normalized scale
def compute_llh_all(y_norm, params):
    """
    y_norm: (n_instances, n_obs) — runtimes normalized by y_max
    params: (n_instances, 2) — predicted [scale(floc), sigma] on normalized scale
    Returns mean log-pdf across all entries
    """
    scale = params[:, 0][:, None]
    sigma = params[:, 1][:, None]
    logy  = np.log(y_norm)
    logsc = np.log(scale)
    exponent = -0.5 * ((logy - logsc) ** 2) / (sigma ** 2)
    logpdf = exponent - np.log(sigma * np.sqrt(2 * np.pi) * y_norm)
    return np.mean(logpdf)

# 6) Plot Figure 4
plt.figure(figsize=(6, 4))
for scen in scenarios:
    rts  = true_rts[scen]  # (n_instances, n_obs)
    idxs = np.arange(rts.shape[0])
    means, stds = [], []

    for k in sample_sizes:
        fold_vals = []
        for fold, (train_idx, valid_idx) in enumerate(kf.split(idxs)):
            # determine pickle filename
            if k == 100:
                fn = f"{scen}.{task}.{model_name}.{fold}.{seed}.pkl"
            else:
                fn = f"{scen}.{task}.{model_name}.{fold}.{seed}_{k}.pkl"
            pkl_path = os.path.join(base_dir, scen, f"k{k}", fn)
            print("Loading:", pkl_path)
            if not os.path.isfile(pkl_path):
                raise FileNotFoundError(f"Missing pickle: {pkl_path}")
            _, val_pred, _ = pickle.load(open(pkl_path, 'rb'))

            # compute normalization factor y_max from train
            y_train = rts[train_idx]    # shape (n_train_inst, n_obs)
            y_max   = y_train.max()

            # normalize validation runtimes
            y_val    = rts[valid_idx]    # (n_valid_inst, n_obs)
            y_norm   = y_val / y_max

            # val_pred is (n_valid_inst, 2)
            params   = val_pred
            fold_vals.append(compute_llh_all(y_norm, params))

        means.append(np.mean(fold_vals))
        stds.append(np.std(fold_vals))

    plt.errorbar(
        sample_sizes,
        means,
        yerr=stds,
        marker='o',
        capsize=3,
        label=scen
    )

plt.xscale('log', base=2)
plt.xlabel('Number of training samples per instance (k)')
plt.ylabel('Test log-likelihood (Eq. 4)')
plt.title('Figure 4 – DistNet (lognormal)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 7) Save & show
plt.savefig('figure4_lognormal.png', dpi=300)
plt.show()
