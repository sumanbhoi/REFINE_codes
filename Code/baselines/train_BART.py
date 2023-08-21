import dill
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import jaccard_score
from bartpy.sklearnmodel import SklearnModel
import os

import sys
sys.path.append('..')
from util import multi_label_metric

np.random.seed(1203)
model_name = 'BART'

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

def create_dataset(data, diag_voc, med_voc, lab_voc):
    i1_len = len(diag_voc.idx2word)
    i2_len = len(lab_voc.idx2word)
    output_len = len(med_voc.idx2word)
    input_len = i1_len + i2_len
    X = []
    y = []
    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[3]
            o = visit[1]
            # print(i1)
            # print(i2)
            # print(o)

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

    return np.array(X), np.array(y)


def main():
    grid_search = False

    data_path = '../../Data/records_mimiciv_morethan2visit_withNDCmapping_24hrmed_icd9toicd10.pkl'
    voc_path = '../../Data/voc_mimiciv_morethan2visit_withNDCmapping_24hrmed_icd9toicd10.pkl'

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, med_voc, lab_voc = voc['diag_voc'], voc['med_voc'], voc['lab_voc']

    med_rep = med_voc.word2idx


    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point+eval_len:]
    data_test = data[split_point:split_point + eval_len]
    # print(data_test[837])

    train_X, train_y = create_dataset(data_train, diag_voc, med_voc, lab_voc)
    test_X, test_y = create_dataset(data_test, diag_voc, med_voc, lab_voc)
    eval_X, eval_y = create_dataset(data_eval, diag_voc, med_voc, lab_voc)

    # # print(med_voc.idx2word)
    # # print(test_y, np.shape(test_y))
    # if grid_search:
    #     params = {
    #         'estimator__penalty': ['l2'],
    #         'estimator__C': np.linspace(0.00002, 1, 100)
    #     }
    #
    #     model = LogisticRegression()
    #     classifier = OneVsRestClassifier(model)
    #     lr_gs = GridSearchCV(classifier, params, verbose=1).fit(train_X, train_y)
    #
    #     print("Best Params", lr_gs.best_params_)
    #     print("Best Score", lr_gs.best_score_)
    #
    #     return


    model = SklearnModel()  # Use default parameters
    classifier = OneVsRestClassifier(model)
    classifier.fit(eval_X[0:5], eval_y[0:5])  # Fit the model
    print('done')
    # # predictions = model.predict()  # Make predictions on the train set
    y_pred = classifier.predict(test_X[0:5])
    print(out_of_sample_predictions)
    print(test_y[0:5])
    y_prob = classifier.predict_proba(test_X[0:5])

    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(test_y[0:5], y_pred, y_prob)

    # ddi rate
    ddi_A = dill.load(open('../Data/ddi_adj_m.pkl', 'rb'))
    all_cnt = 0
    dd_val = 0
    med_cnt = 0
    visit_cnt = 0
    for adm in y_pred:
        med_code_set = np.where(adm == 1)[0]
        visit_cnt += 1
        med_cnt += len(med_code_set)
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                dd_val = dd_val + ddi_A[med_i, med_j]
                # if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                #     dd_cnt += 1
    wddi_rate = dd_val / all_cnt
    print('\tDDI Rate: %.4f, Jaccard: %.4f, PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        wddi_rate, ja, prauc, avg_p, avg_r, avg_f1
    ))

    history = defaultdict(list)
    for i in range(2):
        history['jaccard'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)

    dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

    print('avg med', med_cnt / visit_cnt)


if __name__ == '__main__':
    main()