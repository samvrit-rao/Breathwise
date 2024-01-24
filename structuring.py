import pandas as pd
import requests
import os
import openai

input = pd.read_csv("/Users/raosamvr/Downloads/mimic-iii-clinical-database-1.4/INPUTEVENTS_MV.csv")
notes = pd.read_csv("/Users/raosamvr/Downloads/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv")
labs = pd.read_csv('/Users/raosamvr/Downloads/mimic-iii-clinical-database-1.4/LABEVENTS.csv')
diagnosis = pd.read_csv('/Users/raosamvr/Downloads/D_ICD_DIAGNOSES.csv')
actual_events = pd.read_csv('/Users/raosamvr/Downloads/DIAGNOSES_ICD(1).csv')

x=1

df = notes.groupby('SUBJECT_ID')['TEXT'].apply(' '.join).reset_index()
fev1 = df['TEXT'].str.contains('FEV1', case=False, na=False).sum()
rales = df['TEXT'].str.contains('rales', case=False, na=False).sum()
rhonchi = df['TEXT'].str.contains('rhonchi', case=False, na=False).sum()

mask_rales = df['TEXT'].str.contains('rales', case=False, na=False)
mask_rhonchi = df['TEXT'].str.contains('rhonchi', case=False, na=False)
mask_wheezing = df['TEXT'].str.contains('wheezing', case=False, na=False)
mask_crackles = df['TEXT'].str.contains('crackles', case=False, na=False)

mask_breath_sounds = mask_rales | mask_rhonchi | mask_wheezing | mask_crackles

mask_FEV1 = df['TEXT'].str.contains('FEV1', case=False, na=False)
mask_exacerbation = df['TEXT'].str.contains('exacerbation', case=False, na=False)
mask_COPD_Asthma = df['TEXT'].str.contains('COPD|Asthma', case=False, regex=True, na=False)

final_mask = mask_breath_sounds & mask_FEV1 & mask_COPD_Asthma
new_df = df[final_mask]
x=1
lab_to_item_codes = {
    'Eosinophils': [51368, 51419, 51444, 51114, 51199, 51120, 51474],
    'SerumPH': [50820, 50831],
    'CReactiveProtein': [50889],
    'PaCO2': [50818, 50830],
    'PO2': [50821, 50832],
    'Oxygen': [50816],
    'Oxygen Saturation': [50817],
    'WBC': [51300, 51301]
}

lab_results_df = pd.DataFrame(columns=['SUBJECT_ID'] + list(lab_to_item_codes.keys()))

for subject_id in df['SUBJECT_ID'].unique():
    subject_labs = labs[labs['SUBJECT_ID'] == subject_id]
    subject_values = {'SUBJECT_ID': subject_id}
    
    for lab_test, item_codes in lab_to_item_codes.items():
        matching_values = subject_labs[subject_labs['ITEMID'].isin(item_codes)]['VALUENUM']
        matching_values_numeric = pd.to_numeric(matching_values, errors='coerce')
        subject_values[lab_test] = matching_values_numeric.mean() if not matching_values_numeric.empty else -1
    
    subject_df = pd.DataFrame([subject_values])
    lab_results_df = pd.concat([lab_results_df, subject_df], ignore_index=True)

final_df = pd.merge(new_df, lab_results_df, on='SUBJECT_ID', how='left')


conditions_icd_codes = {
    'hypertension': ['4010', '4011', '4019'],
    'heart_failure': ['4280', '4281', '42821', '42822', '42823', '42830', '42831', '42832', '42833', '42840', '42841', '42842', '42843', '4289', '4290', '4291', '4292'],
    'diabetes_type_two': ['V180', '25000', '25002', '25010', '25012', '25020', '25022', '25030', '25032', '25040', '25042', '25050', '25052', '25060', '25062', '25070', '25080', '25072', '25082', '25090', '25092'],
}

actualfinaldf = pd.DataFrame(columns=['SUBJECT_ID'] + list(conditions_icd_codes.keys()))

for subject_id in actual_events['SUBJECT_ID'].unique():
    subject_events = actual_events[actual_events['SUBJECT_ID'] == subject_id]

    subject_conditions = {'SUBJECT_ID': subject_id}
    
    for condition, icd_codes in conditions_icd_codes.items():
        match = any(subject_events['ICD9_CODE'].isin(icd_codes))
        subject_conditions[condition] = 1 if match else 0

    actualfinaldf = pd.concat([actualfinaldf, pd.DataFrame([subject_conditions])], ignore_index=True)

final_merged_df = pd.merge(final_df, actualfinaldf, on='SUBJECT_ID', how='left')



x=1
