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
  ├── gat # still in progress
  ├── lgbm
  ├── tabpfn # still in progress
  └── xgboost
```
Models that should really only be run on GPU are:
- Chemeleon (DL)
- Chemprop (DL)
- TabPFN (huge featurizer)

  
  
# `runs/`
Directory that contains recipe.yaml files and model plots & metric results of note. The structure is as follows:

``` 
runs/
├── {target_name}/
  ├── {model_type}
    ├── {YYYYMMDD_initials_anvil_training}
  └── {model_type}
    └── {YYYYMMDD_initials_anvil_training}
```