

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

df = pd.read_excel('/Users/raosamvr/Downloads/modified_file_corrected4.xlsx')
df.fillna(-1, inplace=True)

encoder = OneHotEncoder()
clusters_encoded = encoder.fit_transform(df[['cluster_labels']]).toarray()
cluster_cols = encoder.get_feature_names_out(['cluster_labels'])
df[cluster_cols] = clusters_encoded

feature_cols = ['Exacerbations', 'COPD vs Asthma', 'Age', 'Gender Identification', 'Breath Sounds',
                'FEV1 Value', 'Smoking', 'Eosinophils', 'SerumPH', 'CReactiveProtein', 'PaCO2',
                'PO2', 'Oxygen', 'Oxygen Saturation', 'WBC', 'hypertension', 'heart_failure',
                'diabetes_type_two', 'cardiovascular'] 
target_cols = ['Inhaled Corticosteroids', 'Short-Acting Beta Agonists', 'Long-Acting Beta Agonists',
               'Anticholinergics', 'Phosphodiesterase-4 Inhibitors', 'Mucolytics', 'Oral Steroids',
               'Bronchodilators']

df_filtered = df[df['Exacerbations'] == 0]
df_exacerbations = df[df['Exacerbations'] == 1]
X_exacerbations = df_exacerbations[feature_cols]
X_filtered = df_filtered[feature_cols]

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

best_models = {}


def plot_roc_curve(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

df_filtered=df
X_filtered = df_filtered[feature_cols]

for target in target_cols:
    y_filtered = df_filtered[target]
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)

    if len(y_train.unique()) == 2:
        print(f"Evaluating {target}:")
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")

        accuracies = cross_val_score(best_model, X_train, y_train, cv=skf, scoring='accuracy')
        print("Accuracies per fold:", accuracies)
        print("Average Accuracy:", accuracies.mean())

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print("AUC score on test set:", auc_score)
        print()
        best_models[target] = best_model

        # Visualizations for each model
        plot_roc_curve(y_test, y_pred_proba, f'ROC Curve for {target}')
        plot_feature_importance(best_model, feature_cols, f'Feature Importance for {target}')



predictions_data = []
for index, row in X_exacerbations.iterrows():
    patient_data = row.to_frame().transpose()
    patient_predictions = [model.predict_proba(patient_data)[:, 1][0] for model in best_models.values()]
    predictions_data.append(patient_predictions)

predictions_df = pd.DataFrame(predictions_data, columns=target_cols)
pred_0 = predictions_df



df_f0 = df_filtered[df_filtered['cluster_labels']==0]
df_f1 = df_filtered[df_filtered['cluster_labels']==1]
df_f2 = df_filtered[df_filtered['cluster_labels']==2]
df_f3 = df_filtered[df_filtered['cluster_labels']==3]

dfs = [df_f0, df_f1, df_f2, df_f3]

for i, df in enumerate(dfs):
    column_sums = df[target_cols].sum()

    plt.figure(figsize=(10, 6))
    column_sums.plot(kind='bar')
    plt.title(f'Sum of Each Column for DataFrame {i} (Cluster {i})')
    plt.xlabel('Columns')
    plt.ylabel('Sum')
    plt.xticks(rotation=45)
    plt.show()