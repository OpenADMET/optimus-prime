# Transfer learning from a primary screen

Two stages. The first trains a ChemProp network on primary-screen
log<sub>2</sub> fold change (log<sub>2</sub>FC). The second freezes it, reads its
predictions as two feature columns, concatenates them with a CheMeleon embedding
reduced to 256 principal components, and fits TabICL to dose-response
pEC<sub>50</sub>.

From the OpenADMET pregnane X receptor (PXR) blind challenge analysis
([pxr-challenge-tabicl](https://github.com/OpenADMET/pxr-challenge-tabicl)), on
the partitions that analysis used.

```
transfer_learning/
├── PXR_log2fc_single_concentration.parquet   stage 1 data, 10,830 compounds
├── PXR_pEC50_fit.parquet                     stage 2 fit, 4,392 compounds
├── PXR_pEC50_test_phase2.parquet             stage 2 test, 260 compounds
├── chemprop_log2fc/                          stage 1
└── tabicl_pec50/                             stage 2
```

## Running it

Order matters, and so does the output directory name: `anvil` appends a
timestamp and hash to a directory that already exists, which would leave stage 2
pointing at nothing.

```bash
cd chemprop_log2fc
openadmet anvil --recipe-path chemprop_log2fc.yaml --output-dir chemprop_log2fc_model

cd ../tabicl_pec50
openadmet anvil --recipe-path tabicl_pec50.yaml
```

Stage 1 wants a GPU. Stage 2 extracts embeddings on the GPU but runs TabICL on
CPU deliberately: at 4,392 rows and 258 columns it does not fit in 24 GiB. About
160 seconds per fit.

## The split

Stage 2 reads `train_resource` and `test_resource`, so the splitter never runs
and the partitions are fixed to the challenge's own: fit on the dose-response
training set pooled with phase 1 (4,392 compounds), score on the blinded phase 2
set alone (260).

A run of these recipes scored **MAE 0.4343** on those 260 compounds, against
**0.4269** for the analysis's five-seed mean at a recorded seed spread of 0.007,
**0.4113** for the leading challenge entry, and roughly **0.50** for the previous
CheMeleon baseline. Every other recipe in this repository uses a random split and
reports an optimistic number; this one does not.

The log<sub>2</sub>FC pool shares 2,728 compounds with the fit partition, which
is the transfer working, and none with phase 2, so no scored compound carries a
feature from a network that saw its labels.

## Features

| Block | Native | After PCA | Why |
| --- | --- | --- | --- |
| `CheMeleonEmbeddingFeaturizer` | 2,048 | 256 | 256, 384 and 512 were statistically tied in the analysis; 128 and below were worse |
| `TrainedModelFeaturizer` | 2 | 2 | Each column means one concentration's log<sub>2</sub>FC, so a PCA would rotate them without narrowing them |

Blocks are keyed and ordered by featurizer class name, so CheMeleon comes first
whatever order the recipe lists.

Two ingredient choices were ties rather than wins. The analysis's nominal best
used a log<sub>2</sub>FC-fine-tuned CheMeleon embedding, tied with the
off-the-shelf one used here and needing a third trained model. TabICL tied
TabPFN v3 (0.427 against 0.436) and is used for its BSD 3-Clause license.

## Data provenance

`PXR_log2fc_single_concentration.parquet` comes from
`pxr-challenge_single_concentration_TRAIN.csv` in the public
[`openadmet/pxr-challenge-train-test`](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
dataset, published in long format. It is filtered to the two populated
concentration arms (10,747 and 9,523 observations, against 706 and 27 for the
pair dropped) and pivoted to one row per compound, with repeats averaged. SMILES
are canonicalized with RDKit after largest-fragment stripping.

The two pEC<sub>50</sub> parquets are converted from `data/splits/fit_all.csv`
and `data/splits/test_phase2.csv` in pxr-challenge-tabicl, reused rather than
rederived so the rows match the published metrics exactly.
