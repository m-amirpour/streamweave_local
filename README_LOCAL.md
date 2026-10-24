# StreamWeave — local Fedora pilot

This package runs the existing StreamWeave research-candidate pipeline locally and has local-first, **offline-only** dataset loading. It is intended to validate the local execution path and run a modest pilot on a laptop. It is **not** a guarantee of publication-ready results.

## 1. Hardware/runtime expectations

The default configuration is a reduced pilot profile (5,000 sampled rows per dataset, 3 training seeds, fewer model/transfer/CU combinations, no drift, batch size 128). It is meant to check feasibility on a 16 GiB laptop. The full original sweep can take substantially longer and use more memory.

PyTorch automatically uses CUDA if `torch.cuda.is_available()` is true; otherwise it falls back to CPU. On Fedora, NVIDIA driver availability alone does not guarantee the installed PyTorch package has CUDA support. Verify with the check below.

## 2. Install dependencies

If internet/package access is available temporarily:

```bash
cd StreamWeave_Local
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU mode")'
```

If this machine cannot access package indexes, download compatible wheels on a machine with access, transfer them to this machine, and install with `python -m pip install --no-index --find-links /path/to/wheels -r requirements.txt`. Torch wheels must match your Python version and desired CPU/CUDA build. Do not assume CUDA is enabled until the check above confirms it.

## 3. Put datasets in `data/`

This runner does not make network requests. See `data/README.txt` for expected names and formats. Expected files:

- `data/covtype.data` or `data/covtype.data.gz`
- `data/Skin_NonSkin.txt` (or `data/skin.csv`)
- `data/KDDTrain+.txt`

You can run only datasets you have locally by editing `DATASETS_TO_RUN` near the top of `streamweave_local.py`. If a dataset is missing, that dataset will report a clear failure and the script will continue to the next one.

## 4. Validate before real data

```bash
python -m py_compile streamweave_local.py smoke_test_local.py
python smoke_test_local.py
```

The smoke test uses synthetic data only and does not download datasets. It is a software sanity check, not scientific evidence.

## 5. Run the pilot

```bash
python streamweave_local.py
```

Results and checkpoints are saved in `StreamWeave_Local_Results/` next to the script. Keep that directory intact to allow resume behavior. Start a new output directory after changing the code or experimental configuration.

## 6. Moving from pilot to paper-scale experiments

Do not report the default pilot as the final experiment. After the pilot runs successfully and you inspect failed-run CSVs, deliberately restore the planned sample size, seed count, architectures, transfer methods, combination units, and drift settings in the config. Record the exact final configuration and rerun with a fresh results directory. Seeds in this implementation vary model/training randomness on one fixed split; they are not independent data splits. The implementation does not include a faithful original HDBS decision-tree baseline, and its drift is synthetic. Do not claim exact reproduction or superiority over the base paper until the protocol and baseline are implemented and checked.
