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


