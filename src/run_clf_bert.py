import os
import sys

mode = sys.argv[1]
n_trials = 10
n_epochs = 4
feats_path = "../data/feat-combined.csv"
data_path = "../data/posts-sents.csv"
domain_adapt = ""
domain = "domain40"
fusion_dim = 0
feats = ""
cmd = f"~/anaconda/bin/python clf_bert.py -mode {mode} -name domain -feat_names {feats} -domain {domain} {domain_adapt} -fusion_dim {fusion_dim} -weight_classes -n_trials {n_trials} -n_epochs {n_epochs} -feats {feats_path} -data {data_path}"
print(cmd)
os.system(cmd)
