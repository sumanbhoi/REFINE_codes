import pandas as pd
import numpy as np
from dask import dataframe as df0

pd.options.mode.chained_assignment = None
# ndc2atc_file = 'ndc2atc_level4.csv'
ndc2atc_file = 'ndc_map 2020_06_17_atc5 atc4 ingredients.csv'
cid_atc = 'drug-atc.csv'
ndc2rxnorm_file = 'ndc2rxnorm_mapping.txt'
icd_conv = pd.read_csv('icd9toicd10cmgem.csv')
icd_conv = icd_conv.drop(columns=['flags', 'approximate', 'no_map', 'combination', 'scenario', 'choice_list'])

code_dict = dict(zip(icd_conv.icd9cm, icd_conv.icd10cm))
df1 = pd.read_csv("labevents.csv", usecols=['subject_id', 'hadm_id', 'itemid', 'valuenum'],
                  dtype={'subject_id': np.int64, 'hadm_id': np.float64, 'itemid': np.int64, 'valuenum': np.float64},
                  low_memory=False)
# print(len(df1['itemid'].unique()))

# # df2 = pd.read_csv("prescriptions.csv", dtype={'ndc':'category'})
df2 = pd.read_csv("prescriptions.csv", usecols=['subject_id', 'hadm_id', 'starttime', 'ndc', 'dose_val_rx'],
                  dtype={'ndc': 'category'})
df3 = pd.read_csv("diagnoses_icd.csv", usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])
# # print('meds', df2.head(10))

print('Data read done')


def uniqueIndexes(l):
    seen = set()
    res = []
    for i, n in enumerate(l):
        if n not in seen:
            res.append(i)
            seen.add(n)
    return res


def process_lab():
    lab_pd = df1
    # lab_pd.drop(columns=['entereddateyyyymmdd1', 'resultvalue1'], inplace=True)
    lab_pd.dropna(inplace=True)
    lab_pd.drop_duplicates(inplace=True)
    # lab_pd.resultitemdesc = lab_pd.resultitemdesc.apply(lambda x: x.replace(',', ''))
    print('lab type', type(lab_pd.valuenum[117]))
    lab_pd['value_norm'] = lab_pd.groupby('itemid')['valuenum'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    lab_pd['value_norm'] = lab_pd['value_norm'].fillna(0)
    lab_pd.sort_values(by=['subject_id', 'hadm_id'], inplace=True)
    #     print(len(lab_pd))
    #     print(lab_pd.head())
    # lab_pd = lab_pd.reset_index(drop=True)
    return lab_pd.reset_index(drop=True)


def med_startdate(df):
    df_new = df.drop(columns=['ndc', 'starttime', 'dose_val_rx'])
    df_new = df_new.groupby(by=['subject_id', 'hadm_id']).head([1]).reset_index(drop=True)
    df_new = pd.merge(df_new, df, on=['subject_id', 'hadm_id', 'Date'])
    df_new['time_diff'] = df_new.groupby('subject_id')['Date'].apply(lambda x: x - x.iloc[0]) / np.timedelta64(1, 'D')
    df_new = df_new.drop(columns=['starttime', 'Date'])
    df_new = df_new.drop_duplicates()
    df_new = df_new.reset_index(drop=True)
    return df_new


def process_med():
    #     med_pd = pd.read_pickle(data_path + "prescriptions")
    med_pd = df2
    print(np.shape(med_pd))
    med_pd.drop(index=med_pd[med_pd['ndc'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    # print('drop na', np.shape(med_pd))
    # print(chunk['starttime'].head(10))
    med_pd['starttime'] = pd.to_datetime(med_pd['starttime'], format='%Y-%m-%d %H:%M:%S')
    med_pd['Date'] = pd.to_datetime(med_pd['starttime']).dt.date
    med_pd.sort_values(by=['subject_id', 'hadm_id', 'Date'], inplace=True)
    med_pd = med_startdate(med_pd)

    med_pd.drop_duplicates(inplace=True)
    med_pd = med_pd[pd.to_numeric(med_pd['dose_val_rx'], errors='coerce').notnull()]
    med_pd['dose_val_rx'] = med_pd['dose_val_rx'].apply(pd.to_numeric, errors='coerce')
    med_pd['dose_val_norm'] = med_pd.groupby('ndc')['dose_val_rx'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    print('null dose value', np.shape(med_pd))
    med_pd = med_pd.reset_index(drop=True)
    return med_pd.reset_index(drop=True)


def ndc2atc4(med_pd):
    print(np.shape(med_pd))
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['rxcui'] = med_pd['ndc'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)
    print(np.shape(med_pd))
    rxnorm2atc = pd.read_csv(ndc2atc_file, dtype={'rxcui': 'category'})
    rxnorm2atc = rxnorm2atc.drop(
        columns=['atc4', 'atc4_name', 'ndc', 'in_rxcui', 'has_min', 'in_tty', 'in_synonym', 'in_umlscui', 'atc5', 'ddd',
                 'u', 'adm_r', 'whocc_note'])
    rxnorm2atc.drop_duplicates(subset=['rxcui'], inplace=True)
    rxnorm2atc = rxnorm2atc.dropna(subset=['rxcui', 'in_name'])
    rxnorm2atc['rxcui'] = rxnorm2atc['rxcui'].astype('int64')
    med_pd.drop(index=med_pd[med_pd['rxcui'].isin([''])].index, axis=0, inplace=True)
    print(np.shape(med_pd))
    med_pd['rxcui'] = med_pd['rxcui'].astype('int64')
    print(type(med_pd['rxcui'][0]))
    med_pd = med_pd.reset_index(drop=True)
    print(type(rxnorm2atc['rxcui'][0]))
    med_pd = med_pd.merge(rxnorm2atc, on=['rxcui'])
    print(np.shape(med_pd))
    med_pd.drop(columns=['ndc', 'rxcui'], inplace=True)
    med_pd = med_pd.rename(columns={'in_name': 'drug_name'})
    #     med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd['drug_name'] = med_pd.drug_name.str.replace("(/).*", "")
    med_pd['drug_name'] = med_pd.drug_name.str.replace("(,).*", "")
    med_pd['drug_name'] = med_pd['drug_name'].str.replace('\d+', '')
    # med_pd['drug_name'] = med_pd['drug_name'].astype('category')
    med_pd = med_pd.drop_duplicates()
    med_pd.dropna(inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    return med_pd


def process_diag():
    #     diag_pd = pd.read_pickle(data_path + "diagnoses_icd")
    diag_pd = df3
    diag_pd.dropna(inplace=True)
    diag_pd['icd_code_10'] = diag_pd['icd_code'].map(code_dict)
    diag_pd.head().to_dict()
    diag_pd.loc[diag_pd['icd_version'] == 10, 'icd_code_10'] = diag_pd['icd_code']
    diag_pd.drop(diag_pd[diag_pd['icd_code_10'] == 'nan'].index, inplace=True)
    diag_pd = diag_pd.drop(columns=['icd_code', 'icd_version'])
    print('diag done')
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['subject_id', 'hadm_id'], inplace=True)
    return diag_pd.reset_index(drop=True)


def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['icd_code_10']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['icd_code_10'].isin(diag_count.loc[:1999, 'icd_code_10'])]

    return diag_pd.reset_index(drop=True)


def process_all():
    # get med and diag (visit>=2)
    lab_pd = process_lab()
    med_pd = process_med()
    # print(med_pd.head(10))
    med_pd = ndc2atc4(med_pd)
    print(type(med_pd.dose_val_rx[2]))
    #     med_pd['dose_val_rx'] = med_pd['dose_val_rx'].apply(pd.to_numeric, errors='coerce')
    #     print(type(med_pd.dose_val_rx[2]))
    #     med_pd['dose_val_norm'] = med_pd.groupby('drug_name')['dose_val_rx'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    # print('After processing', med_pd.head(10))
    # print(type(med_pd['drug_name']))
    # print(type(med_pd['hadm_id']))
    # print(med_pd['drug_name'])
    diag_pd = process_diag()
    diag_pd = filter_2000_most_diag(diag_pd)

    med_pd_key = med_pd[['subject_id', 'hadm_id']].drop_duplicates()
    diag_pd_key = diag_pd[['subject_id', 'hadm_id']].drop_duplicates()
    lab_pd_key = lab_pd[['subject_id', 'hadm_id']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['subject_id', 'hadm_id'], how='inner')
    combined_key = combined_key.merge(lab_pd_key, on=['subject_id', 'hadm_id'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    #     diag_pd_duplicate = diag_pd
    med_pd = med_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    #     pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    lab_pd = lab_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    # med_pd_duplicate = med_pd
    #     lab_pd_duplicate

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['subject_id', 'hadm_id'])['icd_code_10'].unique().reset_index()
    #     diag_pd_duplicate = diag_pd_duplicate.groupby(by=['patientcode','vID'])['icd10cm'].unique().reset_index()
    med_pd = med_pd.drop_duplicates(subset=['subject_id', 'hadm_id', 'drug_name'])

    med_pd = med_pd.groupby(['subject_id', 'hadm_id']).agg(lambda x: list(x))
    print('val in med_pd', med_pd)

    for index, row in med_pd.iterrows():
        # print('in loop')
        print(row)
        # print(row['dose_val_rx'])
        indexes = uniqueIndexes(row['drug_name'])
        #     item = row['ITEMID']
        # print(indexes)
        item = [row['drug_name'][i] for i in indexes]
        # val = [row['dose_val_rx'][i] for i in indexes]
        actual_val = [row['dose_val_rx'][i] for i in indexes]
        val = [row['dose_val_norm'][i] for i in indexes]
        # diff_val = [row['diff'][i] for i in indexes]
        med_pd['drug_name'][index] = item
        # med_pd['dose_val_rx'][index] = val
        med_pd['dose_val_norm'][index] = val
        med_pd['dose_val_rx'][index] = actual_val
        # med_pd['diff'][index] = diff_val

        #     print('final', med_pd)
    lab_pd = lab_pd.drop_duplicates(subset=['subject_id', 'hadm_id', 'itemid'])
    lab_pd = lab_pd.groupby(['subject_id', 'hadm_id']).agg(lambda x: list(x))
    # print('lab', type(lab_pd['itemid'][0]))
    # print(type(lab_pd['hadm_id'][0]))
    # print(type(lab_pd['subject_id'][0]))
    for index, row in lab_pd.iterrows():
        print('in loop lab')
        print(row['itemid'])
        print(row)
        indexes = uniqueIndexes(row['itemid'])
        #     item = row['ITEMID']
        item = [row['itemid'][i] for i in indexes]
        val = [row['value_norm'][i] for i in indexes]
        actual_val = [row['valuenum'][i] for i in indexes]
        lab_pd['itemid'][index] = item
        lab_pd['value_norm'][index] = val
        lab_pd['valuenum'][index] = actual_val

    #     lab_pd = lab_pd.groupby(by=['patientcode','vID'])['resultvalue_nbr'].unique().reset_index()

    data = diag_pd.merge(med_pd, on=['subject_id', 'hadm_id'], how='inner')
    data = data.merge(lab_pd, on=['subject_id', 'hadm_id'], how='inner')

    # visit > 2
    def process_visit2_data(data):
        b = data[['subject_id', 'hadm_id']].groupby(by='subject_id')['hadm_id'].unique().reset_index()
        #         print(a.head())
        b['hadm_id_Len'] = b['hadm_id'].map(lambda x: len(x))
        b = b[b['hadm_id_Len'] >= 2]
        return b

    #
    data_pd_visit2 = process_visit2_data(data).reset_index(drop=True)
    data = data.merge(data_pd_visit2[['subject_id']], on='subject_id', how='inner')

    print('Done')
    return data


data = process_all()
print(data.head(10))
print(data.columns)
data.to_pickle('data_mimiciv_morethan2visit_withNDCmapping_24hrmed_icd9toicd10_dose.pkl')