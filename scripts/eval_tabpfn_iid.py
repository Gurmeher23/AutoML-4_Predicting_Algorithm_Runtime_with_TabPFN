import argparse, os, pickle, sys, pathlib
import numpy as np
import torch
from tabpfn import TabPFNRegressor

# Make DistNet modules importable regardless of CWD
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from helper import load_data, preprocess, data_source_release
from sklearn.model_selection import KFold


def predict_nll(reg,
                X_vl_flat32,
                Y_vl_flat32_transformed,
                batch_size: int = 4096):
    """
    Compute continuous NLL in batches on the transformed (standardized) data.
    """
    n = X_vl_flat32.shape[0]
    out_chunks, pos = [], 0
    eps = 1e-12

    while pos < n:
        end = min(pos + batch_size, n)
        with torch.no_grad():
            full = reg.predict(X_vl_flat32[pos:end], output_type="full")

            criterion = full.get("criterion", None) or getattr(reg, "renormalized_criterion_", None) or getattr(reg, "bardist_", None)
            if criterion is None:
                raise RuntimeError("TabPFN criterion not found. Update tabpfen or keep output_type='full'.")

            dev = criterion.borders.device
            logits = torch.as_tensor(full["logits"], device=dev, dtype=torch.float32)
            y_true_transformed = torch.as_tensor(Y_vl_flat32_transformed[pos:end], device=dev, dtype=torch.float32).view(-1)

            # NLL in transformed (Z-score) space
            pdf_transformed = criterion.pdf(logits, y_true_transformed).clamp_min(eps)
            nll = -torch.log(pdf_transformed)

        out_chunks.append(nll.detach().cpu())
        pos = end

    return torch.cat(out_chunks, dim=0).numpy()


def main():
    p = argparse.ArgumentParser()
    sc_dict = data_source_release.get_sc_dict()
    p.add_argument("--scenario", required=True, choices=sc_dict.keys())
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--save", required=True)

    p.add_argument("--subsample_mode", choices=["first", "flattened_random", "instance_wise"], default="instance_wise",
                   help="Subsampling strategy. 'instance_wise' is recommended for better results.")
    p.add_argument("--cap_rows", type=int, default=4096,
                   help="Max training context rows. For instance_wise, this is used to calculate the number of instances to sample.")
    p.add_argument("--passes", type=int, default=2,
                   help="Repeat iid subsample + re-fit and average NLLs for more stable results.")
    p.add_argument("--val_batch", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1)

    args = p.parse_args()
    assert 0 <= args.fold <= 9

    print(f">> TabPFN (Subsample: {args.subsample_mode}): Standardization (Z-score). NLL in STANDARDIZED space.")

    data_dir = data_source_release.get_data_dir()
    runtimes, features, _ = load_data.get_data(
        scenario=args.scenario, data_dir=data_dir,
        sc_dict=sc_dict, retrieve=sc_dict[args.scenario]["use"])

    X, Y = np.asarray(features), np.asarray(runtimes)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    tr_idx, vl_idx = list(kf.split(np.arange(X.shape[0])))[args.fold]

    X_tr_raw, X_vl_raw = X[tr_idx], X[vl_idx]
    Y_tr_raw, Y_vl_raw = Y[tr_idx], Y[vl_idx]

    X_tr, X_vl = preprocess.preprocess_features(X_tr_raw, X_vl_raw, scal="meanstd")
    n_reps = Y_tr_raw.shape[1]

    X_tr_flat = np.repeat(X_tr, n_reps, axis=0).astype(np.float32)
    Y_tr_flat = Y_tr_raw.flatten().astype(np.float32)
    X_vl_flat = np.repeat(X_vl, n_reps, axis=0).astype(np.float32)
    Y_vl_flat = Y_vl_raw.flatten().astype(np.float32)

    # Standardize targets using Mean and Std Dev from TRAINING data
    train_mean = np.mean(Y_tr_flat)
    train_std = np.std(Y_tr_flat)
    if train_std < 1e-8:
        train_std = 1.0

    Y_tr_flat_std = ((Y_tr_flat - train_mean) / train_std).astype(np.float32)
    Y_vl_flat_std = ((Y_vl_flat - train_mean) / train_std).astype(np.float32)

    print(f"  Train Mean: {train_mean:.4f}, Train Std Dev: {train_std:.4f}")

    nll_passes = []
    for pass_id in range(args.passes):
        rng_pass = np.random.RandomState(args.seed + pass_id)

        if args.subsample_mode == "instance_wise":
            n_instances_tr = X_tr.shape[0]
            n_instances_to_sample = max(1, min(n_instances_tr, args.cap_rows // n_reps))
            print(f"  [Pass {pass_id+1}/{args.passes}] Sampling {n_instances_to_sample}/{n_instances_tr} instances...")

            sub_instance_idx = rng_pass.choice(n_instances_tr, size=n_instances_to_sample, replace=True)

            X_sub_instances = X_tr[sub_instance_idx]
            Y_tr_std = Y_tr_flat_std.reshape(Y_tr_raw.shape)
            Y_sub_instances_std = Y_tr_std[sub_instance_idx]

            X_sub = np.repeat(X_sub_instances, n_reps, axis=0).astype(np.float32)
            Y_sub = Y_sub_instances_std.flatten().astype(np.float32)

        else: # flattened_random or first
            N_flat = X_tr_flat.shape[0]
            cap = min(args.cap_rows, N_flat)
            print(f"  [Pass {pass_id+1}/{args.passes}] Sampling {cap}/{N_flat} rows...")
            sub_idx = rng_pass.choice(N_flat, size=cap, replace=True)
            X_sub, Y_sub = X_tr_flat[sub_idx], Y_tr_flat_std[sub_idx]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        reg = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
        reg.fit(X_sub, Y_sub)

        nll_rep = predict_nll(reg, X_vl_flat, Y_vl_flat_std, batch_size=args.val_batch)
        val_pred = nll_rep.reshape(-1, n_reps).mean(axis=1)
        nll_passes.append(val_pred)

    val_pred_mean = np.mean(np.stack(nll_passes, axis=0), axis=0)

    os.makedirs(args.save, exist_ok=True)
    add_info = {
        "scenario": args.scenario, "fold": args.fold, "model": "tabpfn",
        "cap_rows": args.cap_rows, "passes": args.passes, "subsample_mode": args.subsample_mode,
        "seed": args.seed, "y_transform": "standardize", "report_space": "standardized_zscore_space",
        "scale_meta": {"transform": "zscore", "train_mean": train_mean, "train_std": train_std}
    }
    filename = (f"{args.scenario}.full.tabpfn.sample_{args.subsample_mode}"
                f".cap{args.cap_rows}.passes{args.passes}.fold{args.fold}.rep_standardized_space.pkl")
    fp = os.path.join(args.save, filename)

    with open(fp, "wb") as fh:
        pickle.dump([None, val_pred_mean, add_info], fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {fp} | Mean NLL: {float(np.mean(val_pred_mean)):.4f}")


if __name__ == "__main__":
    main()