
from rank_metrics import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import numpy as np


pivot_metric = "auc"

feat_acr = {"senti_score": "sscr", "senti_polar": "spol",
            "senti_class:pos": "spos", "senti_class:neu": "sneu",
            "senti_class:neg": "sneg", 
            "topic10": "t10", "topic50": "t50", "topic100": "t100",
            "kialo_wo3_freq": "kwo3f",
            "kialo_wo4_freq": "kwo4f", "kialo_wo5_freq": "kwo5f",
            "kialo_wo3_attr": "kwo3a",
            "kialo_wo4_attr": "kwo4a", "kialo_wo5_attr": "kwo5a",
            "kialo_wo3_extreme": "kwo3e",
            "kialo_wo4_extreme": "kwo4e", "kialo_wo5_extreme": "kwo5e",
            "kialo_ukp_avgdist10": "kud",
            "kialo_ukp0.1_attr": "ku1a", "kialo_ukp0.1_extreme": "ku1e",
            "kialo_ukp0.2_attr": "ku2a", "kialo_ukp0.2_extreme": "ku2e",
            "kialo_ukp0.3_attr": "ku3a", "kialo_ukp0.3_extreme": "ku3e",
            "kialo_ukp0.4_attr": "ku4a", "kialo_ukp0.4_extreme": "ku4e",
            "question_confusion": "qcon", "question_whyhow": "qwh",
            "question_other": "qo"}


def accuracy_rank(ys_list):
    """ys_list = [{"y_true": [..], "y_pred": [..]}]"""
    acc = {}
    ndcg, ndcg3, p1, p3, ap, top3 = [], [], [], [], [], []
    for ys in ys_list:
        y_true = np.array(ys["y_true"])
        y_pred = np.array(ys["y_pred"])

        ranked_y_true = y_true[y_pred.argsort()[::-1]]
        doc_len = len(ranked_y_true)
        ndcg.append(ndcg_at_k(ranked_y_true, doc_len))
        ndcg3.append(ndcg_at_k(ranked_y_true, min(3, doc_len)))
        p1.append(precision_at_k(ranked_y_true, min(1, doc_len)))
        p3.append(precision_at_k(ranked_y_true, min(3, doc_len)))
        ap.append(average_precision(ranked_y_true))
        top3.append(int(ranked_y_true[:3].sum() > 0))

    acc = {"ndcg": np.mean(ndcg),
           "ndcg3": np.mean(ndcg3),
           "p1": np.mean(p1),
           "p3": np.mean(p3),
           "map": np.mean(ap),
           "top3": np.mean(top3)}
    
    return acc

def accuracy(y_true, y_prob):
    y_pred = [int(y >= 0.5) for y in y_prob]

    acc = {}
    acc["auc"] = roc_auc_score(y_true, y_prob)
    acc["prec"], acc["recl"], acc["f1"], _ = \
            precision_recall_fscore_support(y_true, y_pred, 
                                            average="binary")

    return acc


