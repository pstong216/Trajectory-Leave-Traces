# Reproducibility notes

## Pin BLAS threads — this is part of the protocol, not an optimisation

Every `random_state` in this codebase is fixed, and that is **not sufficient** on
its own. Floating-point addition is not associative across threads, so a
multi-threaded BLAS can produce slightly different results run to run, which is
occasionally enough to nudge `LogisticRegression`'s `lbfgs` solver to a different
optimum. Seeding controls where the search starts, not how the linear algebra is
computed.

This bit us concretely: one `webarena/claude` run produced AUROC 0.6226 where
five other runs produced 0.7602, purely because numpy was using Apple's
multi-threaded Accelerate BLAS with no thread caps set.

Export these **before** starting the process — BLAS reads the thread count at
library load time, so setting them via `os.environ` inside a script is too late:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42
```

`reproduce.sh` does this for you. Do it by hand for any ad-hoc invocation.

## Environment the released numbers were produced with

```
python 3.13.5   numpy 2.4.4    pandas 2.3.3     scipy 1.16.3
scikit-learn 1.7.2             matplotlib 3.10.8
pyarrow 22.0.0                 lightgbm 4.6.0
```

`requirements.txt` gives lower bounds. Exact agreement to the last decimal is
only expected on the same minor versions — scikit-learn in particular can change
solver internals between releases even with identical seeds.

## What "reproduces" means here

With the thread caps set, reruns are bit-identical, and this was checked against
the published artifacts rather than assumed:

| artifact | check |
|---|---|
| Table 2, Table 5 | generated `.tex` diffed against the paper's own source file |
| Figure 3 | generated PNG byte-identical to the arXiv submission |
| Figures 6, 7, 8 | generated PDFs identical to the arXiv submission except the embedded creation timestamp |
| Figure 5, Figure 9, Tables 7–8 | result CSVs byte-identical across reruns |
| Ours/L2 training | a fresh run reproduced the shipped `summary_table.csv` byte for byte |

Matplotlib stamps a `/CreationDate` into every PDF, so PDF checksums will differ
from run to run. Set `SOURCE_DATE_EPOCH` if you need byte-identical PDFs.
