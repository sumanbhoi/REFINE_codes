from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
from collections import Counter
warnings.filterwarnings('ignore')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# use the same metric from DMNC
def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, train_size=2/3, random_state=1203)
    x_eval, x_test, y_eval, y_test = train_test_split(x_eval, y_eval, test_size=0.5, random_state=1203)
    return x_train, x_eval, x_test, y_train, y_eval, y_test

def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]
    # print('pind',pind)
    out_list = []
    break_flag = False
    # print('length, shape', len(pind), np.shape(pind))
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    # print('out', out_list)
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    # print('sorted predict', sorted_predict)
    ap = np.append([y_pred_prob_tmp], [out_list], axis=0)
    a = sorted(zip(y_pred_prob_tmp, out_list))
    a_list = list(a)
    a1, a2 = zip(*a_list)
    # print('y_pred_leap',y_pred_prob_tmp)
    # print('outlist',out_list)
    # print(list(a1))
    # print((a2))
    ap_sort = ap.sort
    # print('sorted', ap_sort)
    b = np.zeros((1, 153))
    
    for i in range(len(a)):
        b[:, a[i][1]] = a[i][0]
    
    # print('final', b)
    # print("final", sorted_predict)
    return out_list, sorted_predict, b
    # return b


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b]==1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score


    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score


    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)
    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            # print(y_gt.shape[0])
            target = np.where(y_gt[b] == 1)[0]
            # print('target', target)
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    # def f1(y_gt, y_pred):
    #     all_micro = []
    #     for b in range(y_gt.shape[0]):
    #         all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
    #     return np.mean(all_micro)

    # def roc_auc(y_gt, y_prob):
    #     all_micro = []
    #     for b in range(len(y_gt)):
    #         all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
    #     return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        # print('sorted_index', np.shape(sort_index))
        # print('sorted_index', sort_index)
        # print((y_gt))
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    def recall_at_k(y_gt, y_prob, k):
        recall = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        # print('sorted_index', np.shape(sort_index))
        # print('sorted_index', sort_index)
        # print((y_gt))
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            P_instances = np.where(y_gt[i,:] == 1)[0]
            # print('p_instance', len(P_instances))
            recall += TP / len(P_instances)
        return recall / len(y_gt)

    # auc = roc_auc(y_gt, y_prob)
    # p_1 = precision_at_k(y_gt, y_prob, k=1)
    # p_3 = precision_at_k(y_gt, y_prob, k=3)
    # p_5 = precision_at_k(y_gt, y_prob, k=5)
    # p_10 = precision_at_k(y_gt, y_prob, k=10)
    # p_20 = precision_at_k(y_gt, y_prob, k=20)
    # p_30 = precision_at_k(y_gt, y_prob, k=30)
    # # print('gt', len(y_gt))
    # p_k = [p_1, p_3, p_5, p_10, p_20, p_30]
    # p_k = []

    # r_1 = recall_at_k(y_gt, y_prob, k=1)
    # r_3 = recall_at_k(y_gt, y_prob, k=3)
    # r_5 = recall_at_k(y_gt, y_prob, k=5)
    # r_10 = recall_at_k(y_gt, y_prob, k=10)
    # r_20 = recall_at_k(y_gt, y_prob, k=20)
    # r_30 = recall_at_k(y_gt, y_prob, k=30)
    # r_k = [r_1, r_3, r_5, r_10, r_20, r_30]
    # r_k = []
    # f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_pred)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(record, path='../data/ddi_A_final.pkl'):
    # ddi rate
    ddi_A = dill.load(open(path, 'rb'))
    # print('size of ddi_a', np.shape(ddi_A))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt

def wddi_rate_score(record, path='../Data/ddi_adj_m_neurips.pkl'):
    # ddi rate
    ddi_A = dill.load(open(path, 'rb'))
    # print('size of ddi_a', np.shape(ddi_A))
    all_cnt = 0
    dd_cnt = 0
    val = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] != 1 or ddi_A[med_j, med_i] != 1:
                        val = val + ddi_A[med_i, med_j]
    if all_cnt == 0:
        return 0
    return val / all_cnt

# def ddi_rate_score_analysis(record, record_extra, path='../data/ddi_A_final.pkl'):
#     # ddi rate
#     ddi_A = dill.load(open(path, 'rb'))
#     # print('size of ddi_a', np.shape(ddi_A))
#     all_cnt = 0
#     dd_cnt = 0
#     dd_cnt_extra = 0
#     for patient in record:
#         for adm in patient:
#             med_code_set = adm
#             for i, med_i in enumerate(med_code_set):
#                 for j, med_j in enumerate(med_code_set):
#                     if j <= i:
#                         continue
#                     all_cnt += 1
#                     if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
#                         dd_cnt += 1
#
#     for patient in record_extra:
#         for adm in patient:
#             med_code_set = adm
#             for i, med_i in enumerate(med_code_set):
#                 for j, med_j in enumerate(med_code_set):
#                     if j <= i:
#                         continue
#                     # all_cnt += 1
#                     if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
#                         dd_cnt_extra += 1
#
#
#     contribution_extra = (dd_cnt_extra/dd_cnt)*100
#
#     if all_cnt == 0:
#         return 0
#     return contribution_extra

#
# def ddi_rate_score_analysis_missed(GT, missed, extra, path='../data/ddi_A_final.pkl'):
#     # ddi rate
#     ddi_A = dill.load(open(path, 'rb'))
#     all_cnt = 0
#     dd_cnt = 0
#     dd_cnt_extra = 0
#     dd_cnt_missed_inter = 0
#
#     for num, patient in enumerate(GT):
#         for adm_num, adm in enumerate(patient):
#             med_code_set = adm
#             missed_code_set = missed[num][adm_num]
#             extra_code_set = extra[num][adm_num]
#             for i, med_i in enumerate(med_code_set):
#                 for j, med_j in enumerate(med_code_set):
#                     if j <= i:
#                         continue
#                     # all_cnt += 1
#                     if med_i in missed_code_set or med_j in missed_code_set:
#                         dd_cnt += 1
#                         if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
#                             dd_cnt_missed_inter += 1
#
#
#
#     contribution_missed = (dd_cnt_missed_inter / dd_cnt) * 100
#     if dd_cnt == 0:
#         return 0
#     return contribution_missed
#
# def missed_extra_analysis(missed, extra, voc_path = '../data/voc_final.pkl'):
#     voc = dill.load(open(voc_path, 'rb'))
#     diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
#     # med_rep = med_voc.word2idx
#     count = 0
#     missed_count = 0
#     extra_count = 0
#
#     for num, patient in enumerate(missed):
#         for adm_num, adm in enumerate(patient):
#             # med_code_set = adm
#             missed_code_set = adm
#             extra_code_set = extra[num][adm_num]
#             extra_count += len(extra_code_set)
#             for i, med_i in enumerate(missed_code_set):
#                 missed_count += 1
#                 missed_med = med_voc.idx2word.get(med_i)
#                 for j, med_j in enumerate(extra_code_set):
#                     extra_med = med_voc.idx2word.get(med_j)
#                     if missed_med[0:3] == extra_med[0:3]:
#                         count += 1
#
#     replacement_missed = count/(missed_count + extra_count - count)
#     return replacement_missed, (count/1058)

