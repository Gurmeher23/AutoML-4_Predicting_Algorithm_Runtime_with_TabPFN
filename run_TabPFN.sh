

SCENARIOS = [
    "clasp_factoring",
    "spear_qcp",
    "yalsat_qcp",
    "lpg-zeno",
]

SAVE_ROOT  = "results_iid"   # scenario subfolders will be created
CAP_ROWS   = 4096            # iid context size per fold (increase if VRAM allows)
PASSES     = 2               # independent iid subsamples to average
SUBSAMPLE  = "random"        # i.i.d. subsampling (with replacement handled inside the script)
VAL_BATCH  = 4096            # reduce if you hit OOM
SEED       = 1

import subprocess, os, sys, time

os.makedirs(SAVE_ROOT, exist_ok=True)
t_all = time.time()

for sc in SCENARIOS:
    save_dir = os.path.join(SAVE_ROOT, sc)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n=== Scenario: {sc} ===")
    t0 = time.time()

    for fold in range(10):
        print(f"→ fold {fold}")
        cmd = [
            sys.executable, "DistNet/scripts/eval_tabpfn_iid.py",
            "--scenario", sc,
            "--fold", str(fold),
            "--save", save_dir,
            "--cap_rows", str(CAP_ROWS),
            "--passes", str(PASSES),
            "--subsample_mode", SUBSAMPLE,   # default is 'random', kept explicit
            "--val_batch", str(VAL_BATCH),
            "--seed", str(SEED),
            # no --y_transform / --report_space flags needed (defaults handle it)
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"✗ fold {fold} failed (exit {e.returncode}); continuing…")

    print(f"✔ finished {sc} in {(time.time()-t0)/60:.1f} min")

print(f"\n✔ all scenarios done in {(time.time()-t_all)/3600:.2f} h")