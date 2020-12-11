# TASK 1
Use the R script `analy_attackability_char.R`.

Analyze what characteristics of sentences are associated with a sentence’s attackability. (Chapter 5. Task 1: Attackability Characteristics)
Return odds ratio (OR) and statistical significance of features.

# TASK 2
## Logistic Regression
You may start from `run_clf_lr.py`. Modify some parameters as you need and run it:
```
$ python run_clf_lr.py [TASK] [MODE]
```

`[TASK]` 
* `all`: Run with all features with different combinations of LR settings.
* `ablation`: Ablation tests for the best setting.
* `ngram`: Run with only ngrams.

`[MODE]`
* `attack`: Predict attacked vs. unattacked.
* `success`: Predict successfully attacked vs. not.

## BERT classifier
You may start from `run_clf_bert.py`. Modify some parameters as you need and run it:
```
$ python run_clf_bert.py [MODE]
```

`[MODE]`
* `attack`: Predict attacked vs. unattacked.
* `success`: Predict successfully attacked vs. not.

Required libraries: `pytorch`, `pytorch_transformers`, `tqdm`


