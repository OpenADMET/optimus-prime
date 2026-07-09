<div style="text-align: left">
<img src="optimus.png" alt="lil robot guy" width="200"/>  
</div>

# Canonical recipes 
Directory that contains all the prime examples of Anvil recipe.yamls for each model in the OpenADMET suite.  

Current available models are:
``` 
canonical_recipes/
├── multitask/
  ├── chemeleon
  └── chemprop
├── single_task/
  ├── chemeleon
  ├── chemprop
  ├── lgbm
  ├── tabpfn # still in progress
  └── xgboost
  └── catboost
  └── random forest
  └── dummy
```
Models that should really only be run on GPU are:
- Chemeleon (DL)
- Chemprop (DL)
- TabPFN (huge featurizer)

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


