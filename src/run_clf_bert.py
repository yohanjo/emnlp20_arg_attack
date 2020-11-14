import os
import sys

mode = sys.argv[1]  # "attack" for predicting attacked vs. unattacked
                    # "success" for predicting successfully attacked vs. not
n_trials = 10  # Num of total runs
n_epochs = 4  # Num of epochs for each run
feats_path = "../data/feat-combined.csv"  # Feature file
data_path = "../data/posts-sents.csv"  # Sentence file
domain_adapt = ""  # Do not modify
domain = "domain40"  # Do not modify
fusion_dim = 0  # Do not modify
feats = ""  # Features to include (do not modify)
cmd = f"~/anaconda/bin/python clf_bert.py -mode {mode} -name domain -feat_names {feats} -domain {domain} {domain_adapt} -fusion_dim {fusion_dim} -weight_classes -n_trials {n_trials} -n_epochs {n_epochs} -feats {feats_path} -data {data_path}"
print(cmd)
os.system(cmd)
