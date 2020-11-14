import os
import sys
from itertools import product

task = sys.argv[1]  # "all" for all features
                    # "ablation" for ablation tests
                    # "ngram" for ngrams only
mode = sys.argv[2]  # "attack" for predicting attacked vs. unattacked
                    # "success" for predicting successfully attacked vs not

subj_feats = ["senti_score", "senti_class:pos", "senti_class:neu", 
              "senti_class:neg", "arousal", "dominance", "concreteness", 
              "subjectivity", "hedging", "quantification"]
proptype_feats = ["question_confusion", "question_whyhow", "question_other", 
                  "normative", "if", "prediction", "source", "my", "you", 
                  "we", "comparative", "example", "definition"]


# All features
if task == "all":
    n_trials = 10  # Num of runs
    n_epochs = 30  # Num of epochs for each run
    feats_path = "../data/feat-combined.csv"  # Feature file
    data_path = "../data/posts-sents.csv"  # Text file
    frames_path = "../data/sling/posts-nlp.sling.csv"  # Not used
    domain = "domain40"  # Do not modify
    domain_adapt = ""  # Do not modify
    all_feats = ["text", "topic50", "kialo_wo5_freq kialo_wo5_attr kialo_wo5_extreme",
                 " ".join(subj_feats), " ".join(proptype_feats)]
    for penalty, alpha in product(["l1", "l2"], [0.0001, 0.001, 0.01, 0.1]):
        feat_str = " ".join(all_feats)
        cmd = f"~/anaconda/bin/python clf_lr.py -mode {mode} -name feat_comb -feat_names {feat_str} -domain {domain} {domain_adapt} -weight_classes -penalty {penalty} -alpha {alpha} -n_trials {n_trials} -n_epochs {n_epochs} -feats {feats_path} -data {data_path} -frames {frames_path}"
        print(cmd)
        os.system(cmd)


# Ablation
if task == "ablation":
    n_trials = 10  # Num of runs
    n_epochs = 30  # Num of epochs for each run
    feats_path = "../data/feat-combined.csv"  # Feature file
    data_path = "../data/posts-sents.csv"  # Text file
    frames_path = "../data/sling/posts-nlp.sling.csv"  # Not used
    domain = "domain40"  # Do not modify
    domain_adapt = ""  # Do not modify
    penalty = "l2"  # Regularization type for logistic regression
    alpha = 0.1  # Regularization weight for logistic regression
    all_feats = ["text topic50", "kialo_wo5_freq kialo_wo5_attr kialo_wo5_extreme",
                 " ".join(subj_feats), " ".join(proptype_feats)]
    for excl_feat in all_feats:
        feat_str = " ".join([feat for feat in all_feats if feat != excl_feat])
        cmd = f"~/anaconda/bin/python clf_lr.py -mode {mode} -name ablation -feat_names {feat_str} -domain {domain} {domain_adapt} -weight_classes -penalty {penalty} -alpha {alpha} -n_trials {n_trials} -n_epochs {n_epochs} -feats {feats_path} -data {data_path} -frames {frames_path}"
        print(cmd)
        os.system(cmd)


# Top ngrams analysis 
if task == "ngram":
    n_trials = 1  # Num of runs
    n_epochs = 30  # Num of epochs for each run
    feats_path = "../data/feat-combined.csv"  # Feature file
    data_path = "../data/posts-sents.csv"  # Text file
    frames_path = "../data/sling/posts-nlp.sling.csv"  # Not used
    domain = "domain40"  # Do not modify
    domain_adapt = ""  # Do not modify
    feat_str = "text"
    for penalty, alpha in product(["l1", "l2"], [0.0001, 0.001, 0.01, 0.1]):
        cmd = f"~/anaconda/bin/python clf_lr.py -mode {mode} -name ngram -feat_names {feat_str} -domain {domain} {domain_adapt} -weight_classes -penalty {penalty} -alpha {alpha} -n_trials {n_trials} -n_epochs {n_epochs} -feats {feats_path} -data {data_path} -frames {frames_path}"
        print(cmd)
        os.system(cmd)


