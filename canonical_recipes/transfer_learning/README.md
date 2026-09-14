# Transfer learning from a primary screen

A two-stage recipe that reuses an abundant, cheap assay readout to sharpen
predictions on a scarce, expensive one. Stage 1 trains a ChemProp network on
single-concentration primary-screen log<sub>2</sub> fold change
(log<sub>2</sub>FC). Stage 2 freezes that network, reads its predictions as two
feature columns, concatenates them with a CheMeleon embedding reduced by
principal component analysis (PCA), and fits TabICL to dose-response
pEC<sub>50</sub>.

The configuration comes from the OpenADMET pregnane X receptor (PXR) blind
challenge analysis, and these recipes fit and score on the partitions that
analysis used, so their numbers are comparable to the ones it published. Full
write-up and supporting code:
[pxr-challenge-tabicl](https://github.com/OpenADMET/pxr-challenge-tabicl).

```
transfer_learning/
├── PXR_log2fc_single_concentration.parquet   stage 1 data, 10,830 compounds
├── PXR_pEC50_fit.parquet                     stage 2 fit, 4,392 compounds
├── PXR_pEC50_test_phase2.parquet             stage 2 test, 260 compounds
├── chemprop_log2fc/                          stage 1: the auxiliary encoder
└── tabicl_pec50/                             stage 2: the dose-response model
```

## Running it

Order matters, and so does the output directory name. Stage 2 refers to the
stage 1 model by path, and `anvil` appends a timestamp and hash to an output
directory that already exists, which would leave stage 2 pointing at nothing.

```bash
cd chemprop_log2fc
openadmet anvil --recipe-path chemprop_log2fc.yaml --output-dir chemprop_log2fc_model

cd ../tabicl_pec50
openadmet anvil --recipe-path tabicl_pec50.yaml
```

Stage 1 trains a message-passing network and wants a graphics processing unit
(GPU). Stage 2 extracts CheMeleon embeddings on the GPU but runs TabICL on the
central processing unit (CPU) deliberately: at 4,392 rows and 258 columns TabICL
asks for 9.5 GiB on top of the 14 GiB it already holds, which does not fit in
24 GiB. On CPU a fit takes about 160 seconds.

## What each stage does

**Stage 1**, `chemprop_log2fc/chemprop_log2fc.yaml`, is a two-task regression.
The challenge screened four concentrations; the two retained here cover 10,747
and 9,523 of the 10,830 compounds, against 706 and 27 for the pair left out.
Each arm is its own task, so `n_tasks: 2` and `dropna: False`: 1,390 compounds
were screened at only one concentration, and ChemProp masks a missing target out
of the loss rather than discarding the row's other arm.

This network is never scored on its own. It exists so stage 2 can read it, which
is why the recipe carries no evaluation block and no test split.

It trains on every compound in the pool, with noam calibrated to the full
30-epoch budget and early stopping off. See [what this does not
reproduce](#what-these-recipes-do-not-reproduce) for how that relates to the
analysis's two-pass encoder.

**Stage 2**, `tabicl_pec50/tabicl_pec50.yaml`, builds 258 feature columns:

| Block | Native width | After PCA | Why |
| --- | --- | --- | --- |
| `CheMeleonEmbeddingFeaturizer` | 2,048 | 256 | Reductions to 256, 384 and 512 were statistically tied in the analysis, and 256 is the cheapest of them. At 128 and below the reduction was clearly separated and worse. |
| `TrainedModelFeaturizer` | 2 | 2 (passthrough) | Each column means one concentration's predicted log<sub>2</sub>FC. A PCA would rotate them into linear combinations, and at full rank would not reduce the width at all. |

Passthrough is stated as `TrainedModelFeaturizer: null` rather than obtained by
omitting the key, so the exact-match check still catches a mistyped block name.
Blocks are keyed and ordered by featurizer class name, not by the order the
recipe lists them, which is why the CheMeleon block comes first. Mean imputation
runs ahead of each block's PCA, matching the analysis; neither block should carry
a missing value, so it is a guard rather than a fix.

Two choices here were ties rather than wins, and are worth revisiting if you
adapt this. The analysis's nominal best used a CheMeleon embedding fine-tuned on
log<sub>2</sub>FC, which was statistically tied with the off-the-shelf embedding
used here and would need a third trained model to ship. Three configurations
that tie the leader omit the log<sub>2</sub>FC readout entirely, so the case for
keeping stage 1 at all rested on a nominal difference rather than a separated
one. TabICL was likewise tied with TabPFN v3 (seed means 0.427 and 0.436) and is
used here for its BSD 3-Clause license against TabPFN's attribution requirement.

## The split, and what to expect

Stage 2 reads `train_resource` and `test_resource` rather than splitting one
file, so the splitter never runs and the partitions are fixed. This is the
challenge's own evaluation split: fit on the dose-response training set pooled
with phase 1 (4,392 compounds), score on the blinded phase 2 set alone (260).

That makes the reported metric directly comparable to published numbers. The
analysis measured a mean absolute error (MAE) of **0.436** for this
configuration at a single seed, against **0.4113** for the leading challenge
entry and roughly **0.50** for the previous CheMeleon baseline. A run of these
recipes should land near 0.436.

This departs from every other recipe in this repository, which use a random
`ShuffleSplitter` and report optimistic metrics. Here the test compounds are a
genuine blind holdout, so the number means what it says.

The log<sub>2</sub>FC pool and the fit partition share 2,728 compounds, which is
the transfer working as intended. **The pool and the phase 2 test set share
none**, so no scored compound carries a feature from a network that saw any of
its labels. The build script asserts both properties and refuses to write if
either breaks.

## What these recipes do not reproduce

**Five seeds.** The analysis trains five encoders and five regressors and reports
both a seed mean and an ensemble of the five. These recipes run one of each, to
show the method rather than to restate the measurement. The ensemble row scored
0.432 against the 0.436 single-seed mean quoted above.

**Refit-on-all, exactly.** The analysis holds out 20% of the log<sub>2</sub>FC
pool, early-stops against it, then reinitializes and retrains on the full pool
for the epoch count that produced. Anvil derives noam's decay from the trainer's
`max_epochs`, so duration and schedule are one knob and the second pass cannot
run a shorter budget without also recalibrating the schedule. Stage 1 therefore
trains the full 30 epochs on the whole pool: same data, same schedule, and no
fifth of the screen spent finding an epoch count. The analysis chose 30 to sit
close to where early stopping was expected to land, so the two should end up near
each other, though we have not measured the difference.

## Data provenance

`PXR_log2fc_single_concentration.parquet` is built from
`pxr-challenge_single_concentration_TRAIN.csv` in the public
[`openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
dataset, which is published in long format at one row per compound and
concentration. It is filtered to the two populated arms and pivoted to one row
per compound, with repeat measurements of a compound at one concentration
averaged. SMILES are canonicalized with RDKit after largest-fragment stripping.

`PXR_pEC50_fit.parquet` and `PXR_pEC50_test_phase2.parquet` are converted from
`data/splits/fit_all.csv` and `data/splits/test_phase2.csv` in the
pxr-challenge-tabicl repository, which tracks the split it derived from the same
Hugging Face dataset. They are reused rather than rederived so the partitions are
identical to the ones the published metrics came from, down to the row. Those
files carry the same canonical SMILES column, computed the same way.
