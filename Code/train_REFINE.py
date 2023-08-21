import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import linregress

from models import REFINE
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'REFINE'
resume_name = ''
# resume_name = 'Epoch_38_JA_0.5029_DDI_0.0788.model'   # with loss 0.9 and 0.1

def slope_cal(v, t, voc):
    s = np.zeros(len(voc.idx2word))
    if len(v) > 1:
        v_stack = np.vstack(v)
        for i in range(np.shape(v_stack)[1]):
            s[i] = linregress(t[:, 0],v_stack[:, i])[0]
    s[np.isnan(s)] = 0
    return s


def var_func(vi):
    n = len(vi)
    mean = sum(vi) / n
    deviations = [(x - mean) ** 2 for x in vi]
    variance = sum(deviations) / n
    return variance


def var_cal(v, voc):
    var = np.zeros(len(voc.idx2word))
    if len(v) > 1:
        v_stack = np.vstack(v)
        for i in range(np.shape(v_stack)[1]):
            #             print(v_stack[:,i])
            #             print(t)
            var[i] = var_func(v_stack[:, i])

    return var

def slope_var_records(records, med_voc, lab_voc):
    record_new = []
    for step, input_p in enumerate(records):
        patient = []
        # print('patient', step)
        lab_inter = []
        med_inter = []
        time_diff = []
        #     med_stack = []
        for idx, adm in enumerate(input_p):
            visit = []
            # print(idx)
            seq_input = input_p[:idx + 1]
            lab_vec = np.zeros(len(lab_voc.idx2word))
            med_vec = np.zeros(len(med_voc.idx2word))
            lab_vec[adm[3]] = adm[4]
            med_vec[adm[1]] = adm[2]
            # lab test and med vectors over visits
            lab_inter.append(lab_vec)
            med_inter.append(med_vec)
            time_diff.append(adm[6])
            slope_m = slope_cal(med_inter, np.array(time_diff), med_voc)
            var_m = var_cal(med_inter, med_voc)
            slope_l = slope_cal(lab_inter, np.array(time_diff), lab_voc)
            var_l = var_cal(lab_inter, lab_voc)
            visit.append(adm[0])
            visit.append(adm[3])
            visit.append(list(lab_vec))
            visit.append(list(slope_l))
            visit.append(list(var_l))
            visit.append(adm[1])
            visit.append(list(med_vec))
            visit.append(list(slope_m))
            visit.append(list(var_m))
            visit.append(adm[7])
            patient.append(visit)
        record_new.append(patient)
    return record_new

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')

    model.eval()
    smm_record = []
    smm_record_gt = []
    ja, prauc, avg_p, avg_r, avg_f1= [[] for _ in range(5)]

    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    med_cnt_gt = 0
    # To get the representation of the supplementary medication codes
    voc_path = '../Data/voc_trial_nuerips.pkl'
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, med_voc, lab_voc = voc['diag_voc'], voc['med_voc'], voc['lab_voc']
    med_rep = med_voc.word2idx

    A_count = 0
    count_2_or_more = 0
    for step, input in enumerate(data_eval):
        A_count = A_count + 1
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []

        y_gt_label = []
        for adm_idx, adm in enumerate(input):
            print('adm_idx', adm_idx)
            print('input', len(input[:adm_idx+1]))
            target_output1 = model(input[:adm_idx+1])
            y_gt_tmp = np.zeros(voc_size[1])
            y_gt_tmp[adm[5]] = 1
            y_gt.append(y_gt_tmp)

            y_gt_label_tmp = adm[5]
            y_gt_label.append(sorted(y_gt_label_tmp))

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

            label_gt = np.where(y_gt_tmp == 1)[0]
            med_cnt_gt += len(label_gt)

        smm_record.append(y_pred_label)
        smm_record_gt.append(y_gt_label)
        print('y_gt', y_gt)
        print('y_pred', y_pred)
        print('y_pred_prob', y_pred_prob)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))


    wddi_rate = wddi_rate_score(smm_record)
    # ddi_rate_rare = ddi_rate_score(smm_record_rare)
    wddi_rate_gt = wddi_rate_score(smm_record_gt)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        wddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    print('DDI GT', wddi_rate_gt)
    return wddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)



def main():
    if not os.path.exists(os.path.join("saved_original", model_name)):
        os.makedirs(os.path.join("saved_original", model_name))

    data_path = '../Data/records_trial_neurips.pkl'
    voc_path = '../Data/voc_trial_nuerips.pkl'

    ehr_adj_path = '../Data/dco_adj_m_neurips.pkl'
    ddi_adj_path = '../Data/ddi_adj_m_neurips.pkl'
    device = torch.device('cpu:0')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    print(np.shape(ehr_adj))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data_no_trend = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, med_voc, lab_voc = voc['diag_voc'], voc['med_voc'], voc['lab_voc']
    print('med_voc', len(med_voc.idx2word))
    data = slope_var_records(data_no_trend[0:10],med_voc, lab_voc)



    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 10
    LR = 0.0002
    # TEST = args.eval
    TEST = False
    # Neg_Loss = args.ddi
    Neg_Loss = True
    # DDI_IN_MEM = args.ddi
    DDI_IN_MEM = True
    # TARGET_DDI = 0.05
    # T = 0.5
    # decay_weight = 0.85
    num_heads = 8
    num_layers = 6
    d_ff = 256
    max_seq_length = 1000
    dropout = 0.1

    voc_size = (len(diag_voc.idx2word), len(med_voc.idx2word), len(lab_voc.idx2word))
    print('lab voc',len(lab_voc.idx2word))
# Here pass the parameters for the dimension etc.

    model = REFINE(voc_size, ehr_adj, ddi_adj, emb_dim=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=1000, dropout=0.1, device=device, ddi_in_memory=DDI_IN_MEM)

    if TEST:
        # model.load_state_dict(torch.load(open(resume_name, 'rb')))
        model.load_state_dict(torch.load(open(os.path.join("saved_original", model_name, resume_name), 'rb')))
    model.to(device=device)

    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=LR)

    if TEST:
        # med_dist = patient_distribution(data_test)
        eval(model, data_test, voc_size, 0)
        # analysis(model, data_test, voc_size, med_dist)
        print('in test')
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    loss1_target = np.zeros((1, voc_size[1]))
                    loss1_target[:, adm[5]] = 1
                    loss3_target = np.full((1, voc_size[1]), -1)
                    for idx, item in enumerate(adm[5]):
                        loss3_target[0][idx] = item

                    target_output1, balanced_loss = model(seq_input)

                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    loss = 0.75 * loss1 + 0.07 * loss3 + 0.18 * balanced_loss


                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))


            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved_original', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja



        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved_original', model_name, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()
