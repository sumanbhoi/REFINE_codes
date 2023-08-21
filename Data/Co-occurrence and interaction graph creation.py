import pandas as pd
import numpy as np
import dill

path1 = './DDI data/ddinter_downloads_code_A.csv'
path2 = './DDI data/ddinter_downloads_code_B.csv'
path3 = './DDI data/ddinter_downloads_code_D.csv'
path4 = './DDI data/ddinter_downloads_code_H.csv'
path5 = './DDI data/ddinter_downloads_code_L.csv'
path6 = './DDI data/ddinter_downloads_code_P.csv'
path7 = './DDI data/ddinter_downloads_code_R.csv'
path8 = './DDI data/ddinter_downloads_code_V.csv'

file1 = pd.read_csv(path1)
file2 = pd.read_csv(path2)
file3 = pd.read_csv(path3)
file4 = pd.read_csv(path4)
file5 = pd.read_csv(path5)
file6 = pd.read_csv(path6)
file7 = pd.read_csv(path7)
file8 = pd.read_csv(path8)

DDI = pd.concat([file1, file2, file3, file4, file5, file6, file7, file8])
DDI.drop(columns=['DDInterID_A', 'DDInterID_B'],inplace=True)
DDI['Drug_A'] = DDI['Drug_A'].str.lower()
DDI['Drug_B'] = DDI['Drug_B'].str.lower()
print(DDI.head())
print(np.shape(DDI))

# For MIMIC IV
record_m = dill.load(open("./records_mimiciv_morethan2visit_withNDCmapping_24hrmed_icd9toicd10.pkl", 'rb'))
voc_m = dill.load(open("./voc_mimiciv_morethan2visit_withNDCmapping_24hrmed_icd9toicd10.pkl", 'rb'))

med_voc_m = voc_m['med_voc']
med_voc_size_m = len(med_voc_m.idx2word)

# Severity based drug interaction adjacency matrix for mimic iv

ddi_adj_m = np.zeros((med_voc_size_m, med_voc_size_m))
vocab_med_set_m = list(med_voc_m.word2idx.keys())
print(np.shape(DDI))
# print('DDI list', list(DDI['Drug_A']))
for i in range(med_voc_size_m):
    if med_voc_m.idx2word[i] in list(DDI['Drug_A']) or med_voc_m.idx2word[i] in list(DDI['Drug_B']):
        DDI_set = DDI.loc[DDI['Drug_A'] == med_voc_m.idx2word[i]]
        Interact_set = DDI_set['Drug_B']

        if len(Interact_set) > 1:
            for idx, row in DDI_set.iterrows():

                if row['Drug_B'] in vocab_med_set_m:
                    if row['Level'] == 'Major':
                        ddi_adj_m[i, med_voc_m.word2idx[row['Drug_B']]] = 1
                        ddi_adj_m[med_voc_m.word2idx[row['Drug_B']], i] = 1
                    if row['Level'] == 'Moderate':
                        ddi_adj_m[i, med_voc_m.word2idx[row['Drug_B']]] = 0.66
                        ddi_adj_m[med_voc_m.word2idx[row['Drug_B']], i] = 0.66
                    if row['Level'] == 'Minor':
                        ddi_adj_m[i, med_voc_m.word2idx[row['Drug_B']]] = 0.33
                        ddi_adj_m[med_voc_m.word2idx[row['Drug_B']], i] = 0.33
                    if row['Level'] == 'Unknown':
                        ddi_adj_m[i, med_voc_m.word2idx[row['Drug_B']]] = 0
                        ddi_adj_m[med_voc_m.word2idx[row['Drug_B']], i] = 0

dill.dump(ddi_adj_m, open('ddi_adj_m.pkl', 'wb'))

# Frequency based drug co-occurrence adjacency matrix for mimic iv

dco_adj_m = np.zeros((med_voc_size_m, med_voc_size_m))

for patient in record_m:
    for adm in patient:
        med_code_set = adm[1]
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                dco_adj_m[med_i, med_j] = dco_adj_m[med_i, med_j] + 1
                dco_adj_m[med_j, med_i] = dco_adj_m[med_j, med_i] + 1

# Normalize the dco_adj_m matrix
val_max, val_min = dco_adj_m.max(), dco_adj_m.min()
dco_adj_m = (dco_adj_m-val_min)/(val_max-val_min)

dill.dump(dco_adj_m, open('dco_adj_m.pkl', 'wb'))
