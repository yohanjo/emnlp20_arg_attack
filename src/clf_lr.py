from clf_helper import *
from multiprocessing import Pool, cpu_count
import gc
import json
from csv_utils import *
import re
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np
import argparse
from random import shuffle
from time import strftime, localtime, time
import logging
import pickle
from scipy import sparse
from scipy.sparse import csr_matrix
from nltk.stem import WordNetLemmatizer


def sublist(insts, batch_size=128):
    for pos in range(0, len(insts), batch_size):
        yield insts[pos:(pos + batch_size)]

def get_batches(split2data, args):
    split2batches = defaultdict(list)
    for split, insts in split2data.items():
        for subinsts in sublist(insts):
            batch = {}
            batch["text"] = sparse.vstack([inst["text"] for inst in subinsts])
            batch["frames"] = [] #sparse.vstack([inst["frames"] for inst in subinsts])
            batch["y_true"] = [inst["y_true"] for inst in subinsts]
            batch["pid"] = [inst["pid"] for inst in subinsts]
            batch["sid"] = [inst["sid"] for inst in subinsts]
            batch["feats"] = [inst["feats"] for inst in subinsts]
            batch["domains"] = [inst["domains"] for inst in subinsts]

            split2batches[split].append(batch)

    return split2batches

class Analyzer(object):
    def __init__(self, penalty, weight_classes, args=None):
        self.model = SGDClassifier(loss="log", penalty=penalty)
        self.args = args

    def get_class_weights(self, data, equal_weight=False):
        if equal_weight:
            self.class_weights = (1., 1.)
        else:
            ys = [y for batch in data for y in batch["y_true"]]
            self.class_weights = (1, (len(ys)-sum(ys)) / sum(ys))

    def train(self, train_data, val_data, test_data, epochs, 
              weight_classes=True):
        args = self.args

        self.get_class_weights(train_data, equal_weight=(not weight_classes))

        start_time = time()
        max_val_acc = max_test_acc = None
        for epoch in range(1, epochs+1):
            args.logger.info(f"[Epoch {epoch}]")

            post2ys = defaultdict(lambda: defaultdict(list))
            shuffle(train_data)
            #train_data = train_data[::-1][-20:] # top 20 longest
            for batch in tqdm(train_data):
                sample_weight = [self.class_weights[y] for y in batch["y_true"]]
                self.model.partial_fit(batch["input_vec"], batch["y_true"],
                                       classes=np.array([0, 1]), 
                                       sample_weight=sample_weight)

                y_pred_mat = self.model.predict_proba(batch["input_vec"])
                batch["y_pred"] = y_pred_mat[:, 1]
                for pid, y_true, y_pred in zip(batch["pid"], batch["y_true"], 
                                               batch["y_pred"]):
                    post2ys[pid]["y_true"].append(y_true)
                    post2ys[pid]["y_pred"].append(y_pred)


            # Evaluation
            pids = list(post2ys.keys())
            y_true = [y for pid in pids for y in post2ys[pid]["y_true"]]
            y_pred = [y for pid in pids for y in post2ys[pid]["y_pred"]]
            train_acc = accuracy(y_true, y_pred)  # dict
            train_acc.update(accuracy_rank(post2ys.values()))
            args.logger.info(", ".join(["train_{}={}".format(m, v) \
                                    for m, v in sorted(train_acc.items())]))

            val_acc, val_pid2acc = self.test(val_data, split="val")
            test_acc, test_pid2acc = self.test(test_data)

            if max_val_acc is None or \
                    val_acc[pivot_metric] > max_val_acc[pivot_metric]:
                max_val_acc = val_acc
                max_val_pid2acc = val_pid2acc
                max_test_acc = test_acc
                max_test_pid2acc = test_pid2acc

                for b in val_data + test_data:
                    b["y_pred"] = b["y_pred_tmp"]
                    b["input_contr"] = b["input_contr_tmp"]

            delta_time = time() - start_time
            args.logger.info('delta_time={:.1f}m\n'.format(delta_time / 60))

        args.logger.info(", ".join(["best_val_{}={:.3f}".format(
                                    m, v) for m, v in sorted(max_val_acc.items())]))
        args.logger.info(", ".join(["best_test_{}={:.3f}".format(
                                    m, v) for m, v in sorted(max_test_acc.items())]))

        return max_val_acc, max_test_acc, max_val_pid2acc, max_test_pid2acc

    def test(self, data, split="test"):
        args = self.args

        post2ys = defaultdict(lambda: defaultdict(list))
        for batch in data:
            sample_weight = [self.class_weights[y] for y in batch["y_true"]]
            y_pred_mat = self.model.predict_proba(batch["input_vec"])
            batch["y_pred_tmp"] = y_pred_mat[:, 1]
            for pid, y_true, y_pred in zip(batch["pid"], batch["y_true"], 
                                           batch["y_pred_tmp"]):
                post2ys[pid]["y_true"].append(y_true)
                post2ys[pid]["y_pred"].append(y_pred)

            # Feat contribution
            batch["input_contr_tmp"] = batch["input_vec"].multiply(self.model.coef_)
                                            # (batch_size, input_dim)


        # Evaluation
        pids = list(post2ys.keys())
        y_true = [y for pid in pids for y in post2ys[pid]["y_true"]]
        y_pred = [y for pid in pids for y in post2ys[pid]["y_pred"]]
        test_acc = accuracy(y_true, y_pred)  # dict
        test_acc.update(accuracy_rank(post2ys.values()))
        args.logger.info(", ".join(["{}_{}={}".format(split, m, v) \
                                        for m, v in sorted(test_acc.items())]))

        # Individual posts
        pid2acc = {}
        for pid, ys in post2ys.items():
            pid2acc[pid] = accuracy_rank([ys])
            pid2acc[pid]["n_sentences"] = len(ys["y_true"])

        return test_acc, pid2acc


def get_logger(path): 
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    for handler in [logging.FileHandler(path, mode="w"),
                    logging.StreamHandler()]:
        logger.addHandler(handler)

    return logger

def get_prefix(args):
    prefix = "{}-{}".format(args.mode, strftime("%Y%m%d_%H%M%S", localtime()))
    prefix += "-Mlr"
    prefix += f"-NM{args.name}"
    prefix += "-FT" + "_".join([feat_acr.get(f, f[:3]) for f in args.feat_names])
    prefix += f"-DM{args.domain}"
    prefix += f"-DA{args.domain_adapt}"
    prefix += f"-WC{args.weight_classes}"
    prefix += f"-PN{args.penalty}"
    prefix += f"-AL{args.alpha}"

    return prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default="attack", 
                        choices=["success", "attack"])
    parser.add_argument("-name", default="")
    parser.add_argument("-feat_names", nargs="*", default=[])
    parser.add_argument("-attacked_only", action="store_true")
    parser.add_argument("-domain", nargs="?")
    parser.add_argument("-domain_adapt", action="store_true")
    parser.add_argument("-weight_classes", action="store_true")
    parser.add_argument("-n_trials", default=1, type=int)
    parser.add_argument("-n_epochs", default=5, type=int)
    parser.add_argument("-batch_size", default=32, type=int)

    parser.add_argument("-max_ngram", default=2, type=int)
    parser.add_argument("-ngram_voca_size", default=20000, type=int)
    parser.add_argument("-frame_voca_size", default=30000, type=int)

    parser.add_argument("-pickle", action="store_true")
    parser.add_argument("-feats", default="../data/feat-combined.csv")
    parser.add_argument("-data", default="../data/posts-nlp.csv")
    parser.add_argument("-frames", default="../data/sling/posts-nlp.sling.csv")
    parser.add_argument("-pickle_dir", default="../data/pickle")

    parser.add_argument("-penalty", default="l2")
    parser.add_argument("-alpha", type=float, default=0.0001)
    args = parser.parse_args()

    if args.domain_adapt and args.domain is None:
        raise ValueError("Domain adaptation cannot be performed without a domain")

    start_time = time()

    prefix = get_prefix(args)
    print("Prefix:", prefix)

    os.makedirs("../logs", exist_ok=True)
    #log_fnames = os.listdir("../logs")
    #if any(fname.startswith(args.mode) and \
    #        re.search("-Mlr.*", prefix).group() in fname for fname in log_fnames):
    #    print("Already exists. Skip.")
    #    sys.exit()

    args.logger = get_logger(f"../logs/{prefix}.txt")


    if not args.pickle:
        print("Loading features...")
        cnts = Counter()
        split2data = defaultdict(list)
        sid2inst = {}
        pids_include = set()
        header = []
        for r, row in tqdm(enumerate(iter_csv_header(args.feats, 
                                                     header=header))):
            #if cnts[row["split"]] >= 1000: continue
            if r == 0:
                feat_header = [key for key in header \
                    if key.startswith(":") and not key.startswith(":domain")]
                domain_header = [key for key in header \
                                        if key.startswith(":domain")]

            if args.attacked_only and row["direct"] == "0" and row["all_4"] == "0":
                continue

            cnts[row["split"]] += 1
            print(f"\r{r}", end="")

            pid, sid = row["post_id"], int(row["sentence_no"])
            y_true = int(row["success_direct"]) or int(row["success_all_4"]) \
                         if args.mode == "success" \
                         else int(row["direct"]) or int(row["all_4"])

            # Features
            feats = [float(row[key]) for key in feat_header]

            # Domains
            domains = [int(row[key]) for key in domain_header]

            inst = {"pid": row["post_id"],
                    "sid": int(row["sentence_no"]),
                    "y_true": y_true,
                    "feats": feats,
                    "domains": domains,
                    "split": row["split"],
                    "text": "",
                    "frames": ""
            }
            sid2inst[(pid, sid)] = inst
            split2data[row["split"]].append(inst)
            if y_true == 1: pids_include.add(pid)

        # Filter (for memory release)
        for split in split2data.keys():
            split2data[split] = [d for d in split2data[split] \
                                    if d["pid"] in pids_include]
        for pid, sid in list(sid2inst.keys()):
            if pid not in pids_include:
                del sid2inst[(pid, sid)]

        print("\n" + ", ".join([f"{split}={len(split2data[split])}" \
                                for split in ["train", "val", "test"]]))


        #print("Loading frames...")
        #lemmatizer = WordNetLemmatizer()
        #for r, row in tqdm(enumerate(iter_csv_header(args.frames))):
        #    key = (row["post_id"], int(row["sentence_no"]))
        #    if key not in sid2inst: continue

        #    frames = json.loads(row["frames"])
        #    frame_strs = []
        #    for frame in frames:
        #        pred = lemmatizer.lemmatize(frame["PRED"].lower(), pos="v")
        #        fargs = ["_".join([lemmatizer.lemmatize(word.lower()) \
        #                            for word in val.split(" ")])
        #                for farg, val in frame.items() if farg.startswith("ARG")]

        #        for farg in fargs:
        #            frame_strs.append(f"{pred}::{farg}")
        #    
        #    sid2inst[key].update({
        #        "frames": " ".join(frame_strs)
        #    })

        #print("Frame tfidf...")
        #tfidfer = TfidfVectorizer(ngram_range=(1,1), max_features=20000, 
        #                          token_pattern="\\S+")
        #tfidfer.fit([inst["frames"] for sid, inst in \
        #                    sid2inst.items() if inst["split"] == "train"])
        #for sid, inst in sid2inst.items():
        #    inst["frames"] = tfidfer.transform([inst["frames"]])
        #frame2idx = tfidfer.vocabulary_
        frame2idx = {}
        

        print("Loading sentences...")
        for r, row in tqdm(enumerate(iter_csv_header(args.data))):
            key = (row["post_id"], int(row["sentence_no"]))
            if key not in sid2inst: continue
            
            sid2inst[key].update({
                "text": row["sentence_token"]
            })

        print("Text tfidf...")
        tfidfer = TfidfVectorizer(ngram_range=(1,3), max_features=20000, 
                                  token_pattern="\\S+")
        tfidfer.fit([inst["text"] for sid, inst in \
                            sid2inst.items() if inst["split"] == "train"])
        for sid, inst in sid2inst.items():
            inst["text"] = tfidfer.transform([inst["text"]])
        ngram2idx = tfidfer.vocabulary_

        print("\nBuilding batches...")
        split2batches = get_batches(split2data, args)

        if not args.attacked_only:
            os.makedirs(args.pickle_dir, exist_ok=True)
            pickle.dump([split2batches, feat_header, domain_header, 
                         ngram2idx, frame2idx], open(
                    f"{args.pickle_dir}/split2batches-lr-{args.mode}.p", "wb"))

    else:
        print("Loading pickle...")
        split2batches, feat_header, domain_header, ngram2idx, frame2idx = pickle.load(open(
                f"{args.pickle_dir}/split2batches-lr-{args.mode}.p", "rb"))

    ngram_voca = ["_".join(ngram.split(" ")) for ngram, idx in sorted(ngram2idx.items(), key=lambda e: e[1])]
    frame_voca = ["_".join(frame.split(" ")) for frame, idx in sorted(frame2idx.items(), key=lambda e: e[1])]

    args.logger.info("n_posts: " + ", ".join(["{}={}".format(
        split, len(set([pid for b in split2batches[split] for pid in b["pid"]]))) \
                for split in ["train", "val", "test"]]))
    args.logger.info("n_sents: " + ", ".join(["{}={}".format(
                split, sum([len(b["pid"]) for b in split2batches[split]])) \
                        for split in ["train", "val", "test"]]))
    args.logger.info("n_batches: " + ", ".join([f"{split}={len(split2batches[split])}" \
                                        for split in ["train", "val", "test"]]))

    # Filter features and domains
    print("Filtering features and domains...")
    feat_idxs = set()
    for f, feat_name in enumerate(feat_header):
        for feat_prefix in args.feat_names:
            if feat_name.startswith(":"+feat_prefix+":"):
                feat_idxs.add(f)
                break
    feat_voca = [feat_header[f] for f in sorted(feat_idxs)]

    domain_idxs = [] if not args.domain else \
                  set([d for d, domain_name in enumerate(domain_header) \
                        if domain_name.startswith(":"+args.domain+":")])
    domain_voca = [domain_header[d] for d in sorted(domain_idxs)]


    feat_dim = len(feat_idxs)
    domain_dim = len(domain_idxs)
    args.logger.info(f"feat_dim={feat_dim}, domain_dim={domain_dim}")

    for split, batches in split2batches.items():
        for b in batches:
            mats_to_concat = []
            input_voca = []
            if "text" in args.feat_names:
                mats_to_concat.append(b["text"])
                input_voca.extend(ngram_voca)
            if "frames" in args.feat_names:
                mats_to_concat.append(b["frames"])
                input_voca.extend(frame_voca)
            if feat_dim:
                feats = np.array([[val for f, val in enumerate(row) \
                            if f in feat_idxs] for row in b["feats"]])
                mats_to_concat.append(csr_matrix(feats))
                input_voca.extend(feat_voca)
            if domain_dim:
                domain = np.array([[mask for d, mask in enumerate(row) \
                            if d in domain_idxs] for row in b["domains"]])
                mats_to_concat.append(csr_matrix(domain))
                input_voca.extend(domain_voca)
            if args.domain_adapt:
                assert feat_dim and domain_dim
                inter = np.zeros((feats.shape[0], feat_dim * domain_dim))
                didxs = domain.argmax(axis=1)
                for r in range(feats.shape[0]):
                    inter[r, (didxs[r] * feat_dim):((didxs[r]+1) * feat_dim)] = feats[r]
                mats_to_concat.append(csr_matrix(inter))
                input_voca.extend([f"{f}x{d}" for d, f in product(domain_voca, feat_voca)])

            if len(mats_to_concat) == 1:
                b["input_vec"] = mats_to_concat[0].tocsr()
            else:
                b["input_vec"] = sparse.hstack(mats_to_concat).tocsr()

            del b["text"], b["frames"], b["feats"], b["domains"]
    gc.collect()

    print("Analysis...")
    val_accs, test_accs = defaultdict(list), defaultdict(list)
    split2pid2acc = {}
    for trial in range(args.n_trials):
        args.logger.info(f"=================== TRIAL {trial+1} ====================")
        analyzer = Analyzer(args.penalty, args.weight_classes, args)
        val_acc, test_acc, split2pid2acc["val"], split2pid2acc["test"] = \
                analyzer.train(split2batches["train"], split2batches["val"], 
                               split2batches["test"], args.n_epochs)

        for m, v in val_acc.items():
            val_accs[m].append(v)
        for m, v in test_acc.items():
            test_accs[m].append(v)

    args.logger.info(f"=====================================================")
    args.logger.info(", ".join(["final_val_{}={:.3f} ({:.3f})".format(m, np.mean(v), np.std(v)) \
                                for m, v in val_accs.items()]))
    args.logger.info(", ".join(["final_test_{}={:.3f} ({:.3f})".format(m, np.mean(v), np.std(v)) \
                                for m, v in test_accs.items()]))


    args.logger.info("======================================================")
    if "text" in args.feat_names:
        ngram_voca_np = np.array(ngram_voca, dtype=str)
        weights = analyzer.model.coef_[0][:len(ngram2idx)]
        sorted_idxs = weights.argsort()[::-1]
        top_feats = ngram_voca_np[sorted_idxs[:100]]
        bottom_feats = ngram_voca_np[sorted_idxs[-100:]][::-1]
        args.logger.info(f"Top ngrams: {' '.join([f.replace(' ', '_') for f in top_feats])}")
        args.logger.info(f"Bottom ngrams: {' '.join([f.replace(' ', '_') for f in bottom_feats])}")
        
    if "frames" in args.feat_names:
        frame_voca_np = np.array(frame_voca, dtype=str)
        start = len(ngram2idx) if "text" in args.feat_names else 0
        weights = analyzer.model.coef_[0][start:(start + len(frame2idx))]
        sorted_idxs = weights.argsort()[::-1]
        top_feats = frame_voca_np[sorted_idxs[:100]]
        bottom_feats = frame_voca_np[sorted_idxs[-100:]][::-1]
        args.logger.info(f"Top frames: {' '.join(top_feats)}")
        args.logger.info(f"Bottom frames: {' '.join(bottom_feats)}")
        
    if any(re.search("^topic\\d+$", f) for f in args.feat_names):
        start = 0
        if "text" in args.feat_names: start += len(ngram2idx)
        if "frames" in args.feat_names: start += len(frame2idx)

        n_topics = None
        for f in args.feat_names:
            m = re.search("^topic(\\d+)$", f)
            if m: 
                n_topics = int(m.group(1))
                break

        weights = analyzer.model.coef_[0][start:(start + n_topics)]
        sorted_idxs = weights.argsort()[::-1]
        args.logger.info(f"Top topics: {' '.join([str(i) for i in sorted_idxs[:10]])}")
        args.logger.info(f"Bottom topics: {' '.join([str(i) for i in sorted_idxs[-10:][::-1]])}")
 

    # Save dev/test results
    print("Printing insts result...")
    input_voca_np = np.array(input_voca, dtype=str)
    for split, batches in split2batches.items():
        if split == "train": continue
        post2sent2ys = defaultdict(dict)
        for b in batches:
            if "y_pred" not in b: continue
            for pid, sid, y_true, y_pred, input_contr in \
                    zip(b["pid"], b["sid"], b["y_true"], b["y_pred"], b["input_contr"].toarray()):
                sorted_idxs = input_contr.argsort()[::-1]
                high_feats = ", ".join(["{} ({:.3f})".format(input_voca[i], input_contr[i]) \
                                            for i in sorted_idxs[:50] if input_contr[i] > 0])
                low_feats = ", ".join(["{} ({:.3f})".format(input_voca[i], input_contr[i]) \
                                            for i in sorted_idxs[-50:][::-1] if input_contr[i] < 0])
                post2sent2ys[pid][sid] = (y_true, y_pred, high_feats, low_feats)

        with open(f"../logs/{prefix}-insts-{split}.csv", "w") as f:
            out_csv = csv.writer(f)
            out_csv.writerow(["post_id", "sentence_no", "y_true", "y_pred", "high_feats", "low_feats"])
            for pid, sid2ys in post2sent2ys.items():
                for sid, (y_true, y_pred, high_feats, low_feats) in sorted(sid2ys.items()):
                    out_csv.writerow([pid, sid, y_true, y_pred, high_feats, low_feats])

        with open(f"../logs/{prefix}-posts-{split}.csv", "w") as f:
            print_metrics = sorted(list(split2pid2acc[split].values())[0].keys() - set(["n_sentences"]))
            out_csv = csv.writer(f)
            out_csv.writerow(["post_id"] + ["n_sentences"] + print_metrics)
            for pid, acc in split2pid2acc[split].items():
                out_csv.writerow([pid, acc["n_sentences"]] + [acc[key] for key in print_metrics])
    

    args.logger.info('Total time: {:.1f}m\n'.format((time() - start_time) / 60))


