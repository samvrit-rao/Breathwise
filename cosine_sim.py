import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/Users/raosamvr/Downloads/modified_file_corrected4.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)
df.fillna(-1, inplace=True)

df['index'] = range(len(df))

# Define treatment columns
treatment_columns = ['Inhaled Corticosteroids', 'Short-Acting Beta Agonists', 
                     'Long-Acting Beta Agonists', 'Anticholinergics', 
                     'Phosphodiesterase-4 Inhibitors', 'Mucolytics', 
                     'Oral Steroids', 'Bronchodilators']  

df.drop(['FileName'], inplace=True, axis=1)

# Create a new column for combined treatment plans
df['combined_treatment'] = df[treatment_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)

exacerbation_df = df[df['Exacerbations'] == 1]

results = []

# Loop through each patient with exacerbations in the DataFrame
for index, patient in exacerbation_df.iterrows():
    current_cluster_label = patient['cluster_labels']
    current_features = patient.drop(['Exacerbations', 'combined_treatment', 'index'])  
    
    no_exacerbations_same_cluster = df[(df['Exacerbations'] == 0) & (df['cluster_labels'] == current_cluster_label)]
    
    # Extract the features of patients with no exacerbations in the same cluster
    features = no_exacerbations_same_cluster.drop(['Exacerbations', 'combined_treatment', 'index'], axis=1)
    
    cosine_sim = cosine_similarity([current_features], features)

    current_treatment = patient['combined_treatment']
    treatment_plans = no_exacerbations_same_cluster['combined_treatment']
    
    vectorizer = CountVectorizer()
    treatment_matrix = vectorizer.fit_transform(treatment_plans)
    
    treatment_cosine_sim = cosine_similarity(vectorizer.transform([current_treatment]), treatment_matrix)
    
    weighted_treatment_cosine_sim = treatment_cosine_sim
    
    cosine_sim += weighted_treatment_cosine_sim

    most_similar_patient_index = cosine_sim.argmax()
    
    most_similar_patient = no_exacerbations_same_cluster.iloc[most_similar_patient_index]

    similarity_score = cosine_sim.max()

    result = {
        'Patient Index (Exacerbations)': index,
        'Cluster Label (Exacerbations)': current_cluster_label,
        'Current Patient Features (Exacerbations)': current_features,
        'Current Patient Treatment Plan (Exacerbations)': current_treatment,
        'Most Similar Patient Index (No Exacerbations)': most_similar_patient['index'],
        'Most Similar Patient Features (No Exacerbations)': most_similar_patient.drop(['Exacerbations', 'combined_treatment', 'index']),
        'Most Similar Patient Cluster (No Exacerbations)': current_cluster_label,
        'Most Similar Patient Treatment Plan (No Exacerbations)': most_similar_patient['combined_treatment'],
        'Cosine Similarity Score': similarity_score
    }
    results.append(result)

    print("Patient", index, "in Cluster", current_cluster_label, "with exacerbations has the highest cosine similarity to:")
    print("Current Patient Features (Exacerbations):", current_features)
    print("Current Patient Treatment Plan (Exacerbations):", current_treatment)
    print("Most Similar Patient Features (No Exacerbations):", most_similar_patient.drop(['Exacerbations', 'combined_treatment', 'index']))
    print("Most Similar Patient Treatment Plan (No Exacerbations):", most_similar_patient['combined_treatment'])
    print("Cosine Similarity Score:", similarity_score)
    print()

results_df = pd.DataFrame(results)
results_df.to_csv('/Users/raosamvr/Downloads/similar_patients_results_with_exacerbations_noweight.csv', index=False)


new = pd.read_csv('/Users/raosamvr/Downloads/similar_patients_results_with_exacerbations.csv')
cosine_similarity_scores = new['Cosine Similarity Score'].tolist()

for index, similarity_score in enumerate(cosine_similarity_scores):
    if similarity_score > 1:
        cosine_similarity_scores[index] = similarity_score / 2


plt.rcParams['font.family'] = 'Arial'
plt.hist(cosine_similarity_scores, bins=20, edgecolor='k')
plt.xlabel('Cosine Similarity Score', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Cosine Similarity Distribution', fontweight='bold')
plt.savefig('/Users/raosamvr/Downloads/cosine.png', dpi=600)
plt.show()

print("Results saved to 'similar_patients_results_with_exacerbations.csv'")
