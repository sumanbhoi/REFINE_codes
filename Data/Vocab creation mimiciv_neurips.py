# Vocab and record creation

# Vocabulary Creation MIMIC-IV
import dill
import pandas as pd


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    #     pro_voc = Voc()
    lab_voc = Voc()
    ## only for DMNC
    #     diag_voc.add_sentence(['seperator', 'decoder_point'])
    #     med_voc.add_sentence(['seperator', 'decoder_point'])
    #     pro_voc.add_sentence(['seperator', 'decoder_point'])

    for index, row in df.iterrows():
        diag_voc.add_sentence(row['icd_code_10'])
        med_voc.add_sentence(row['drug_name'])
        # print(row['drug_name'])
    #         pro_voc.add_sentence(row['PRO_CODE'])
        lab_voc.add_sentence(row['itemid'])

    #     dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc}, file=open('voc_lab_test_polyclinic.pkl','wb'))
    # dill.dump(obj={'diag_voc': diag_voc, 'med_voc': med_voc}, file=open('voc_without_lab_test_mimiciv_allvisit_withNDCmapping.pkl', 'wb'))
    dill.dump(obj={'diag_voc': diag_voc, 'med_voc': med_voc, 'lab_voc': lab_voc},
              file=open('voc_trial_nuerips.pkl', 'wb'))
    print(med_voc)
    return diag_voc, med_voc, lab_voc


def create_patient_record(df, diag_voc, med_voc, lab_voc):
    records = []  # (patient, code_kind:3, codes)  code_kind:diag,med,lab
    patient_num = 0
    for subject_id in df['subject_id'].unique():
        item_df = df[df['subject_id'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['icd_code_10']])
            #             admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['drug_name']])
            admission.append(row['dose_val_norm'])
            # admission.append(row['dose_val_rx'])
            admission.append([lab_voc.word2idx[i] for i in row['itemid']])
            admission.append(row['value_norm'])
            admission.append([row['valuenum']])
            admission.append([row['time_diff'][0]])
            admission.append([patient_num])
            #             print(len(row['VALUENUM']))
            #             print(len(row['ITEMID']))
            patient.append(admission)
        records.append(patient)
        patient_num = patient_num + 1
    #         print(records)
    #         break
    #     dill.dump(obj=records, file=open('records_lab_test_polyclinic.pkl', 'wb'))
    # dill.dump(obj=records, file=open('records_without_lab_test_mimiciv_allvisit_withNDCmapping.pkl', 'wb'))
    dill.dump(obj=records, file=open('records_trial_neurips.pkl', 'wb'))
    #     print(records)
    return records


# path='data_with_lab_test_polyclinic.pkl'
# path = 'data_without_lab_test_mimiciv_allvisit_withNDCmapping.pkl'

path = 'data_mimiciv_morethan2visit_withNDCmapping_24hrmed_icd9toicd10_dose.pkl'
df = pd.read_pickle(path)
# df = data

# print(df['drug_name'].head(10))
# df_trial = df[120:160]
# X = ['nan']
# for idx, row in df_trial.iterrows():
#     # print(type(row['drug_name']))
#     for i in range(len(row['drug_name'])):
#         print(idx)
#         print(row['drug_name'])
#         print(row['drug_name'][i])
#         print(type(row['drug_name'][i]))
#
# check_for_nan = df['drug_name'].isnull().sum()
# print(check_for_nan)
diag_voc, med_voc, lab_voc = create_str_token_mapping(df)
records = create_patient_record(df, diag_voc, med_voc, lab_voc)
print(len(lab_voc.idx2word))
print(records[0:2])