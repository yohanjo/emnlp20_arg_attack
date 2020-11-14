from clf_helper import *
import gc
from csv_utils import *
from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np
import argparse
from pytorch_transformers import BertModel, BertTokenizer, AdamW
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import strftime, localtime, time
import logging
import pickle
import os

class Tokenizer(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(self, text):
        text_tokens = ["[CLS]"] + self.tokenizer.tokenize(text)[:510] + \
                      ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        return input_ids, segment_ids, input_mask

def long_tensor(v):
    tensor = torch.LongTensor(v)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    return tensor

def float_tensor(v):
    tensor = torch.FloatTensor(v)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    return tensor

def sublist(insts):
    pos = 0
    while pos < len(insts):
        input_len = len(insts[pos]["input_ids"])
        batch_size = min(int(4096 / input_len), 512)
        yield insts[pos:(pos + batch_size)]
        pos += batch_size

def get_batches(split2data, args):
    split2batches = defaultdict(list)
    pad_id = args.tokenizer.tokenizer.convert_tokens_to_ids(
            [args.tokenizer.tokenizer.pad_token])[0]
    for split, insts in split2data.items():
        insts = sorted(insts, key=lambda inst: len(inst["input_ids"]), 
                        reverse=True)
        for subinsts in sublist(insts):
            batch = {"input_ids": [], "segment_ids": [], 
                     "input_mask": []}
            max_input_len = max([len(i["input_ids"]) for i in subinsts])
            for inst in subinsts:
                input_len = len(inst["input_ids"])
                batch["input_ids"].append(inst["input_ids"] + \
                                    [pad_id] * (max_input_len - input_len))
                batch["segment_ids"].append(inst["segment_ids"] + \
                                    [0] * (max_input_len - input_len))
                batch["input_mask"].append(inst["input_mask"] + \
                                    [0] * (max_input_len - input_len))

            batch["y_true"] = [inst["y_true"] for inst in subinsts]
            batch["pid"] = [inst["pid"] for inst in subinsts]
            batch["sid"] = [inst["sid"] for inst in subinsts]
            batch["feats"] = [inst["feats"] for inst in subinsts]
            batch["domains"] = [inst["domains"] for inst in subinsts]

            split2batches[split].append(batch)

    return split2batches

class BertClassifier(nn.Module):
    def __init__(self, feat_dim, domain_dim, domain_adapt, fusion_dim=0,
                 n_classes=2, dropout=.5):
        super(BertClassifier, self).__init__()

        self.feat_dim = feat_dim
        self.domain_dim = domain_dim
        self.domain_adapt = domain_adapt

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)

        fd_dim = feat_dim + domain_dim + \
                 (feat_dim * domain_dim if domain_adapt else 0)
        assert fusion_dim == 0 or fd_dim > 0

        if fusion_dim > 0:
            self.classifier = nn.Sequential(
                                nn.Linear(self.bert.config.hidden_size + fd_dim, 
                                          fusion_dim),
                                nn.ReLU(),
                                nn.Linear(fusion_dim, n_classes)
            )
        else:
            self.classifier = nn.Linear(self.bert.config.hidden_size + fd_dim, 
                                        n_classes)


    def forward(self, input_ids, segment_ids, input_mask, feats, domain):
        batch_size = len(input_ids)
        text_out = self.bert(long_tensor(input_ids), 
                             long_tensor(segment_ids),
                             long_tensor(input_mask))[1]
        text_out = self.dropout(text_out)

        feats = float_tensor(feats)
        domain = float_tensor(domain)
        if self.domain_adapt:
            inter = torch.zeros((batch_size, self.feat_dim * self.domain_dim),
                                device=torch.device("cuda"))
            domain_idxs = domain.argmax(dim=1)
            for r in range(batch_size):
                inter[r, (domain_idxs[r] * self.feat_dim):((domain_idxs[r]+1) * self.feat_dim)] = feats[r]
            clf_in = torch.cat((text_out, feats, domain, inter), dim=1)
        else:
            clf_in = torch.cat((text_out, feats, domain), dim=1)

        logits = self.classifier(clf_in)
        
        return logits

class BertAnalyzer(object):
    def __init__(self, model, args=None):
        self.model = model
        self.args = args

    def get_class_weights(self, data, equal_weight=False):
        if equal_weight:
            self.class_weights = float_tensor([1, 1])

        else:
            ys = [y for batch in data for y in batch["y_true"]]
            self.class_weights = float_tensor([1, (len(ys)-sum(ys)) / sum(ys)])

    def train(self, train_data, val_data, test_data, epochs, 
              weight_classes=True):
        args = self.args

        self.get_class_weights(train_data, equal_weight=(not weight_classes))
        optimizer = AdamW(self.model.parameters(), lr=args.learn_rate, 
                          eps=args.adam_epsilon)

        self.model.train()
        start_time = time()
        max_val_acc = max_test_acc = None
        for epoch in range(1, epochs+1):
            args.logger.info(f"[Epoch {epoch}]")

            post2ys = defaultdict(lambda: defaultdict(list))
            train_loss = 0
            shuffle(train_data)
            #train_data = train_data[::-1][-20:] # top 20 longest
            for batch in tqdm(train_data):
                #print(f"\r{b}/{len(train_data)}", end="")
                #GPUtil.showUtilization()

                #print("Len:", len(batch["input_ids"]))
                #batch = {key: value[:8] for key, value in batch.items()}
                #print("Len:", len(batch["input_ids"]), len(batch["input_ids"][0]))
                #if b == 10: break

                optimizer.zero_grad()
                logits = self.model(batch["input_ids"], 
                                    batch["segment_ids"],
                                    batch["input_mask"],
                                    batch["feats"],
                                    batch["domains"])

                loss = F.cross_entropy(
                        logits, 
                        long_tensor(batch["y_true"]),
                        weight=self.class_weights,
                        reduction="mean")
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                batch["y_pred"] = F.softmax(logits, dim=1)[:, 1].detach()\
                                    .cpu().numpy()
                for pid, y_true, y_pred in zip(batch["pid"], batch["y_true"], 
                                               batch["y_pred"]):
                    post2ys[pid]["y_true"].append(y_true)
                    post2ys[pid]["y_pred"].append(y_pred)

                torch.cuda.empty_cache()


            # Evaluation
            pids = list(post2ys.keys())
            y_true = [y for pid in pids for y in post2ys[pid]["y_true"]]
            y_pred = [y for pid in pids for y in post2ys[pid]["y_pred"]]
            train_acc = accuracy(y_true, y_pred)  # dict
            train_acc.update(accuracy_rank(post2ys.values()))
            args.logger.info("train_loss={:.6f}, ".format(train_loss) + \
                    ", ".join(["train_{}={}".format(m, v) \
                                    for m, v in sorted(train_acc.items())]))

            val_loss, val_acc = self.test(val_data, split="val")
            test_loss, test_acc = self.test(test_data)

            if max_val_acc is None or \
                    val_acc[pivot_metric] > max_val_acc[pivot_metric]:
                max_val_acc = val_acc
                max_test_acc = test_acc

                for b in val_data: b["y_pred"] = b["y_pred_tmp"]
                for b in test_data: b["y_pred"] = b["y_pred_tmp"]

            delta_time = time() - start_time
            args.logger.info('delta_time={:.1f}m\n'.format(delta_time / 60))

        args.logger.info(", ".join(["best_val_{}={:.3f}".format(
                                    m, v) for m, v in sorted(max_val_acc.items())]))
        args.logger.info(", ".join(["best_test_{}={:.3f}".format(
                                    m, v) for m, v in sorted(max_test_acc.items())]))

        return max_val_acc, max_test_acc

    def test(self, data, split="test"):
        with torch.no_grad():
            return self._test(data, split)

    def _test(self, data, split="test"):
        args = self.args

        self.model.eval()
        total_loss = 0
        post2ys = defaultdict(lambda: defaultdict(list))
        for batch in data:
            logits = self.model(batch["input_ids"],
                                batch["segment_ids"],
                                batch["input_mask"],
                                batch["feats"],
                                batch["domains"])

            loss = F.cross_entropy(
                    logits, 
                    long_tensor(batch["y_true"]),
                    weight=self.class_weights,
                    reduction="mean")
            total_loss += loss.item()

            batch["y_pred_tmp"] = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for pid, y_true, y_pred in zip(batch["pid"], batch["y_true"], 
                                           batch["y_pred_tmp"]):
                post2ys[pid]["y_true"].append(y_true)
                post2ys[pid]["y_pred"].append(y_pred)

        # Evaluation
        pids = list(post2ys.keys())
        y_true = [y for pid in pids for y in post2ys[pid]["y_true"]]
        y_pred = [y for pid in pids for y in post2ys[pid]["y_pred"]]
        test_acc = accuracy(y_true, y_pred)  # dict
        test_acc.update(accuracy_rank(post2ys.values()))
        args.logger.info("{}_loss={:.6f}, ".format(split, total_loss) + \
                ", ".join(["{}_{}={}".format(split, m, v) \
                                            for m, v in sorted(test_acc.items())]))

        return total_loss, test_acc

    def load_best_model(self):
        self.model.load_state_dict(self.best_model_state_dict)

def get_logger(path): 
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    for handler in [logging.FileHandler(path, mode="w"),
                    logging.StreamHandler()]:
        logger.addHandler(handler)

    return logger

def get_prefix(args):
    prefix = "{}-{}".format(args.mode, strftime("%Y%m%d_%H%M%S", localtime()))
    prefix += "-Mbert"
    prefix += f"-NM{args.name}"
    prefix += "-FT" + "_".join([feat_acr.get(f, f[:3]) for f in args.feat_names])
    prefix += f"-DM{args.domain}"
    prefix += f"-DA{args.domain_adapt}"
    prefix += f"-FD{args.fusion_dim}"
    prefix += f"-WC{args.weight_classes}"

    return prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default="attack", 
                        choices=["success", "attack"])
    parser.add_argument("-name", default="")
    parser.add_argument("-feat_names", nargs="*", default=[])
    parser.add_argument("-domain", nargs="?")
    parser.add_argument("-domain_adapt", action="store_true")
    parser.add_argument("-weight_classes", action="store_true")
    parser.add_argument("-fusion_dim", default=0, type=int)
    parser.add_argument("-n_trials", default=1, type=int)
    parser.add_argument("-n_epochs", default=5, type=int)

    parser.add_argument("-pickle", action="store_true")
    parser.add_argument("-feats", default="../data/feat-combined-short.csv")
    parser.add_argument("-data", default="../data/posts-nlp-short.csv")
    parser.add_argument("-pickle_dir", default="../data/pickle")

    parser.add_argument("-learn_rate", default=1e-5)
    parser.add_argument("-adam_epsilon", default=1e-8, type=float)
    args = parser.parse_args()

    prefix = get_prefix(args)
    os.makedirs("../logs", exist_ok=True)
    args.logger = get_logger(f"../logs/{prefix}.txt")
    print("Prefix:", prefix)

    if not args.pickle:
        print("Loading features...")
        cnts = Counter()
        split2data = defaultdict(list)
        sid2inst = {}
        pids_include = set()
        header = []
        for r, row in tqdm(enumerate(iter_csv_header(args.feats, 
                                                     header=header))):
            #if cnts[row["split"]] >= 50: continue
            if r == 0:
                feat_header = [key for key in header \
                    if key.startswith(":") and not key.startswith(":domain")]
                domain_header = [key for key in header \
                                        if key.startswith(":domain")]

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
                    "domains": domains
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

        print("Loading sentences...")
        args.tokenizer = Tokenizer()
        for r, row in tqdm(enumerate(iter_csv_header(args.data))):
            print(f"\r{r}", end="")
            key = (row["post_id"], int(row["sentence_no"]))
            if key not in sid2inst: continue

            input_ids, segment_ids, input_mask = \
                    args.tokenizer.tokenize(row["sentence"])
            sid2inst[key].update({
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "input_mask": input_mask
            })

        print("\nBuilding batches...")
        split2batches = get_batches(split2data, args)

        print(f"Saving pickle to {args.pickle_dir}...")
        os.makedirs(args.pickle_dir, exist_ok=True)
        pickle.dump([split2batches, feat_header, domain_header], open(
                f"{args.pickle_dir}/split2batches-bert-{args.mode}.p", "wb"))
    else:
        print("Loading pickle...")
        split2batches, feat_header, domain_header = pickle.load(open(
                f"{args.pickle_dir}/split2batches-bert-{args.mode}.p", "rb"))

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
    domain_idxs = [] if not args.domain else \
                  set([d for d, domain_name in enumerate(domain_header) \
                        if domain_name.startswith(":"+args.domain+":")])
    print(f"feat_dim={len(feat_idxs)}, domain_dim={len(domain_idxs)}")
    for split, batches in split2batches.items():
        for b in batches:
            b["feats"] = [[val for f, val in enumerate(row) \
                            if f in feat_idxs] for row in b["feats"]]
            b["domains"] = [[mask for d, mask in enumerate(row) \
                            if d in domain_idxs] for row in b["domains"]]
    gc.collect()

    print("Analysis...")
    val_accs, test_accs = defaultdict(list), defaultdict(list)
    for trial in range(args.n_trials):
        args.logger.info(f"=================== TRIAL {trial+1} ====================")
        model = BertClassifier(len(feat_idxs), len(domain_idxs), args.domain_adapt)
        model.to(torch.device("cuda"))
        analyzer = BertAnalyzer(model, args)

        val_acc, test_acc = analyzer.train(split2batches["train"], 
                                           split2batches["val"], 
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



    # Save dev/test results
    print("Printing insts result...")
    for split, batches in split2batches.items():
        if split == "train": continue
        post2sent2ys = defaultdict(dict)
        for b in batches:
            if "y_pred" not in b: continue
            for pid, sid, y_true, y_pred in \
                    zip(b["pid"], b["sid"], b["y_true"], b["y_pred"]):
                post2sent2ys[pid][sid] = (y_true, y_pred)

        with open(f"../logs/{prefix}-insts-{split}.csv", "w") as f:
            out_csv = csv.writer(f)
            out_csv.writerow(["post_id", "sentence_no", "y_true", "y_pred"])
            for pid, sid2ys in post2sent2ys.items():
                for sid, (y_true, y_pred) in sorted(sid2ys.items()):
                    out_csv.writerow([pid, sid, y_true, y_pred])



