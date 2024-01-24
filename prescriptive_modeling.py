import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = pd.read_excel('/Users/raosamvr/Downloads/modified_file_corrected4.xlsx')

prognostic_factors = ['Age', 'Gender Identification', 'Breath Sounds', 'FEV1 Value', 'hypertension',
                      'heart_failure', 'diabetes_type_two', 'cardiovascular',
                      'cluster_labels', 'SerumPH', 'CReactiveProtein', 'PaCO2', 'PO2',
                      'Oxygen', 'Oxygen Saturation', 'WBC', 'Smoking', 'Eosinophils']

treatment_types = ['Inhaled Corticosteroids', 'Short-Acting Beta Agonists',
                   'Long-Acting Beta Agonists', 'Anticholinergics',
                   'Phosphodiesterase-4 Inhibitors', 'Mucolytics', 'Oral Steroids',
                   'Bronchodilators']

matched_data_all = pd.DataFrame()

for treatment in treatment_types:
    treatment_group = data[data[treatment] == 1]
    control_group = data[(data[treatment] == 0) & (data['treatment'] == 0)]

    if not treatment_group.empty and not control_group.empty:
        X_train, _, y_train, _ = train_test_split(
            treatment_group[prognostic_factors], treatment_group[treatment], test_size=0.2, random_state=42
        )
        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(X_train, y_train)

        cos_similarities = cosine_similarity(control_group[prognostic_factors], X_train)
        matched_indices = np.argmax(cos_similarities, axis=1)
        matched_control_group = control_group.iloc[matched_indices]

        matched_data = pd.concat([treatment_group, matched_control_group])
        matched_data_all = pd.concat([matched_data_all, matched_data])

matched_data_all = matched_data_all.sample(frac=1, random_state=42).reset_index(drop=True)
matched_data_all.to_csv('/Users/raosamvr/Downloads/matcheddata.csv')