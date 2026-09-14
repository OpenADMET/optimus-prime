# Transfer learning from a primary screen

A two-stage recipe that reuses an abundant, cheap assay readout to sharpen
predictions on a scarce, expensive one. Stage 1 trains a ChemProp network on
single-concentration primary-screen log<sub>2</sub> fold change
(log<sub>2</sub>FC). Stage 2 freezes that network, reads its predictions as two
feature columns, concatenates them with a CheMeleon embedding reduced by
principal component analysis (PCA), and fits TabICL to dose-response
pEC<sub>50</sub>.

The configuration comes from the OpenADMET pregnane X receptor (PXR) blind
challenge analysis. Full write-up and supporting code:
[pxr-challenge-tabicl](https://github.com/OpenADMET/pxr-challenge-tabicl).

```
transfer_learning/
├── PXR_log2fc_single_concentration.parquet   stage 1 data, 10,830 compounds
├── PXR_pEC50_dose_response.parquet           stage 2 data, 4,652 compounds
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

Both stages want a graphics processing unit (GPU). Stage 1 trains a message-passing
network. Stage 2 extracts CheMeleon embeddings on the GPU, and TabICL at 258
columns over 4,652 rows may exceed a 24 GB card and fall back to the central
processing unit (CPU), which costs minutes rather than seconds per fit. Reducing
`CheMeleonEmbeddingFeaturizer` to 128 components instead of 256 fits in that
budget if you need it to.

## What each stage does

**Stage 1**, `chemprop_log2fc/chemprop_log2fc.yaml`, is a two-task regression.
The challenge screened four concentrations; the two retained here cover 10,747
and 9,523 of the 10,830 compounds, against 706 and 27 for the pair left out.
Each arm is its own task, so `n_tasks: 2` and `dropna: False`: 1,390 compounds
were screened at only one concentration, and ChemProp masks a missing target out
of the loss rather than discarding the row's other arm.

This network is never scored on its own. It exists so stage 2 can read it.

The hyperparameters are those of the PXR analysis, including `scheduler: noam`
where the other ChemProp recipes in this repository use `plateau`. Noam needs a
known epoch budget and gets one from `max_epochs: 30`; early stopping only ends
the run sooner than the schedule expects.

**Stage 2**, `tabicl_pec50/tabicl_pec50.yaml`, builds 258 feature columns:

| Block | Native width | After PCA | Why |
| --- | --- | --- | --- |
| `CheMeleonEmbeddingFeaturizer` | 2,048 | 256 | Reductions to 256, 384 and 512 were statistically tied in the analysis, and 256 is the cheapest of them. At 128 and below the reduction was clearly separated and worse. |
| `TrainedModelFeaturizer` | 2 | 2 (passthrough) | Each column means one concentration's predicted log<sub>2</sub>FC. A PCA would rotate them into linear combinations, and at full rank would not reduce the width at all. |

Passthrough is stated as `TrainedModelFeaturizer: null` rather than obtained by
omitting the key, so the exact-match check still catches a mistyped block name.
Blocks are keyed and ordered by featurizer class name, not by the order the
recipe lists them, which is why the CheMeleon block comes first.

Two choices here were ties rather than wins, and are worth revisiting if you
adapt this. The analysis's nominal best used a CheMeleon embedding fine-tuned on
log<sub>2</sub>FC, which was statistically tied with the off-the-shelf embedding
used here and would need a third trained model to ship. Three configurations
that tie the leader omit the log<sub>2</sub>FC readout entirely, so the case for
keeping stage 1 at all rested on a nominal difference rather than a separated
one. TabICL was likewise tied with TabPFN v3 (seed means 0.427 and 0.436) and is
used here for its BSD 3-Clause license against TabPFN's attribution requirement.

## Data provenance

Both parquets are built from the public
[`openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
dataset. SMILES are canonicalized with RDKit after largest-fragment stripping,
so the two tables join on structure rather than on whichever string a source
file carried.

`PXR_log2fc_single_concentration.parquet` comes from
`pxr-challenge_single_concentration_TRAIN.csv`, which is published in long
format at one row per compound and concentration. It is filtered to the two
populated arms and pivoted to one row per compound, with repeat measurements of
a compound at one concentration averaged. Result: 10,830 compounds, NaN where a
compound was not screened at that concentration.

`PXR_pEC50_dose_response.parquet` pools `pxr-challenge_TRAIN.csv` with both
unblinded phase files. A compound appearing in more than one file is averaged in
log space, which is the only correct way to average a potency. Result: 4,652
compounds.

## What the numbers mean, and which split produced them

The published result is a mean absolute error (MAE) of roughly 0.43, against
roughly 0.50 for the previous CheMeleon baseline. **That was measured on the
challenge's own split**: fit on the dose-response training set pooled with phase
1 (4,392 compounds), scored on the blinded phase 2 set alone (260 compounds).
The leading challenge entry scored 0.4113 on that same set.

These recipes do not reproduce that split. Like every recipe in this repository
they use `ShuffleSplitter`, a random 70/10/20 holdout over all 4,652 pooled
compounds, so compounds sharing a scaffold can land on both sides. Metrics from
these recipes are an optimistic estimate of generalization to novel chemistry
and are not comparable to the challenge numbers above. Swap in `ScaffoldSplitter`
for a structure-aware estimate; the root [README](../../README.md) covers the
available splitters.

One further caveat specific to this recipe. The log<sub>2</sub>FC pool and the
dose-response set share 2,728 compounds, all of them from the challenge's
training file. Neither unblinded phase overlaps the log<sub>2</sub>FC pool at
all, so against a held-out phase the stage 2 features never saw a test
compound's label of any kind. Under the random split used here, some of those
2,728 land in the test fold carrying a feature from a network trained on their
own log<sub>2</sub>FC measurement. That is a different endpoint from the
pEC<sub>50</sub> being predicted, so it is not target leakage, but it does make
the test metric read more favorably than a clean holdout would. Refit stage 1
inside the fold loop if that is the question being asked.
