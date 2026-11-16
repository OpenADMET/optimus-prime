<div style="text-align: left">
<img src="optimus.png" alt="lil robot guy" width="200"/>
</div>

# Canonical recipes

[![DOI](https://zenodo.org/badge/1032724010.svg)](https://doi.org/10.5281/zenodo.21504162)
[![Validate recipes](https://github.com/OpenADMET/optimus-prime/actions/workflows/validate-recipes.yaml/badge.svg)](https://github.com/OpenADMET/optimus-prime/actions/workflows/validate-recipes.yaml)

Reference [Anvil](https://github.com/OpenADMET/openadmet-models) recipes for
each model in the OpenADMET suite. Each recipe folder has a `.yaml` recipe and
an `.sbatch` script to launch it on a SLURM GPU or CPU node.

```
canonical_recipes/
├── single_task/          CYP3A4 pChEMBL (ChEMBL), data vendored
│   ├── catboost
│   ├── chemeleon
│   ├── chemprop
│   ├── lgbm
│   ├── rf
│   ├── tabpfn
│   └── xgboost
├── multitask/            CYP3A4 / 2D6 / 2C9 / 1A2 log AC50, data vendored
│   ├── chemeleon
│   └── chemprop
├── ensemble/             5-member bagged committees with calibrated uncertainty
│   ├── single_task/      same models as single_task/
│   └── multitask/        same models as multitask/
├── finetuning/
│   └── single_task/chemprop   AChE-pretrained ChemProp fine-tuned on CYP3A4
└── transfer_learning/    two-stage PXR recipe; see its README
    ├── chemprop_log2fc
    └── tabicl_pec50
```

The `ensemble/` recipes reference the same datasets as their non-ensemble
counterparts, but those files are not copied into `ensemble/`. Point
`data.resource` at your own copy before running them.

Run any recipe from its own folder:

```bash
cd canonical_recipes/single_task/xgboost
openadmet anvil --recipe-path xgboost.yaml
```
Models that should really only be run on GPU are:
- Chemeleon (DL)
- Chemprop (DL)
- TabPFN (huge featurizer)
- TabICL (huge featurizer)

`transfer_learning/` is the one recipe folder whose two YAMLs are not
alternatives. They are ordered stages of a single pipeline, and the second reads
the model the first writes. See
[its README](canonical_recipes/transfer_learning/README.md) for the run order.

## Data splitting (random vs. scaffold)

Every recipe in this repo uses a **random** split by default:

```yaml
split:
  type: ShuffleSplitter
  params:
    random_seed: 42
    train_size: 0.7
    val_size: 0.1
    test_size: 0.2
```

`ShuffleSplitter` wraps sklearn's `train_test_split`, so train and test are
drawn from the same random shuffle. Molecules sharing a scaffold can land on
both sides of the split, which lets the model "recognize" near-duplicates of
training compounds at test time. The reported metrics are therefore an
**optimistic** estimate of how the model generalizes to genuinely novel
chemistry. This is a fine default for runnable examples and quick iteration,
but it is *not* the right benchmark for prospective generalization.

### Making a scaffold-split variant

To estimate generalization to unseen chemical series, copy a recipe and change
only the splitter `type` — the `params` (fractions, seed) stay the same:

```yaml
split:
  type: ScaffoldSplitter   # was: ShuffleSplitter
  params:
    random_seed: 42
    train_size: 0.7
    val_size: 0.1
    test_size: 0.2
```

The structure-aware splitters registered in `openadmet.models.split` are:

| Splitter | Behavior |
| --- | --- |
| `ShuffleSplitter` | Random (default; optimistic) |
| `ScaffoldSplitter` | Holds out whole Bemis–Murcko scaffolds — no scaffold appears in both train and test |
| `PerimeterSplitter` | Pushes the most dissimilar pairs into the test set |
| `MaxDissimilaritySplitter` | Test set maximally dissimilar from train |
| `ClusterSplitter` | Splits by chemical cluster |

These operate on the SMILES `input_col`, so they work for any recipe whose
`data.input_col` is a SMILES column.

> **Caveat — cross-validation stays random.** The structure-aware splitters
> apply only to the train/val/test holdout split. The `*RepeatedKFoldCrossValidation`
> evaluators in the `report.eval` block always run a *random* k-fold over the
> full dataset (they are not scaffold-aware, and the splitters log a
> `"... is not available for cross-validation"` warning). So in a scaffold-split
> variant, treat the **holdout test** metrics as the out-of-distribution
> estimate and read the CV metrics as random-split (optimistic) numbers.



## Validating recipes

CI builds every recipe against the latest `openadmet-models` main on each push
and pull request, and weekly, so upstream parameter renames surface here. It
parses each recipe and constructs every component but loads no data and trains
nothing. To run the same check locally in an environment with `openadmet-models`
installed:

```bash
python scripts/validate_recipes.py                     # all recipes
python scripts/validate_recipes.py path/to/recipe.yaml # just one
```
