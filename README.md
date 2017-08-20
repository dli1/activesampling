# activesampling
*activesampling* is a python package offering active sampling method for Large-scale Information Retrieval Evaluation task. This package provides methods for sampling documents to be assessed from a depth-k pool of document collection, and then estimating IR measures (MAP, RP, P@30) of different system runs contributing the pool. 

# Dependencies
* scipy (> 0.19.0)
* numpy (> 1.12.1)
* scikit-learn (> 0.18.1)
* matplotlib (> 2.0.2)
* pandas (> 0.20.1)

# Usage
- Go to the root directory of *activesampling*.
- Run `sample` or `evaluate` via `python -m acsp.scripts.main`.

## Parameters of `acsp.scripts.main`
- action: Choose action, sample: sample documents and estimate measures, evaluate: evaluate model performance.
- experimen: Choose experiment, 1:bias and variance , 2: effectiveness, 3: re-usability.
- split: Whether split data into training/test, no: not split, yes: split.
- split_type: leave-one-group-out or leave-one-run-out.
- trec: TREC Name used in data directory.
- model: Choose models, mtf: move-to-front, importance: importance sampling, mab: multi-armed bandit, activewr: active sampling.
- sample_index: Number indicating the repeated time of sampling and estimating procedure. e.g. 1, 2, 3, ...

Example.

Run active sampling method on TREC-5 dataset via `python -m acsp.scripts.main --action sample --split no --split_type leave-one-group-out --trec TREC-5 --model active --sample_index 0`, then evaluate the sample via `python -m acsp.scripts.main --action evaluate --experiment 1`.

## Result file structure
```
root ------ sample -- TREC-5 -- sample1 -- percentage1 -- mtv.csv, importance.csv, mab.csv, active.csv, ...
                                        -- percentage2 -- ...
                                        -- ...
                                        -- percentage20 -- ...
                             -- sample2 -- ...
                             -- ...
                             -- sample30 
                   -- TREC-6
                   -- ...
    ------ eval -- TREC-5 -- mtf.exp1.csv, mtf.exp2.csv, mtf.exp3.csv, ..., active.exp1.csv,...
                -- ...
```

            
