import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import kmodes
from  kmodes.kprototypes import KPrototypes
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score

df = pd.read_csv('/Users/raosamvr/Downloads/sdfsdf.csv')
df.fillna(-1, inplace=True)
df.drop(columns=['Gender Identification', 'SerumPH', 'PaCO2'], inplace=True)

# Separate categorical and numerical features
categorical_features = ['Breath Sounds', 'hypertension', 'heart_failure', 'diabetes_type_two', 'cardiovascular']
numerical_features = df.columns.difference(categorical_features).tolist()

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])
important_features = ['FEV1 Value', 'Breath Sounds']  


catColumnsPos = [df_scaled.columns.get_loc(col) for col in categorical_features]

num_clusters = 3  
kproto = KPrototypes(n_clusters=num_clusters, init='Cao', n_init=10, verbose=1)

clusters = kproto.fit_predict(df_scaled, categorical=catColumnsPos)
silhouette_avg = silhouette_score(df_scaled, clusters)
print("The average silhouette score is :", silhouette_avg)



df2=pd.read_csv('/Users/raosamvr/Downloads/sdfsdf.csv')
df['cluster_labels'] = clusters
for i in range(num_clusters):
    cluster_data = df[df['cluster_labels'] == i]

    # Exclude -1 from mean calculations
    cluster_data_numerical = cluster_data[numerical_features].replace(-1, np.nan)
    means = cluster_data_numerical.mean()

    print(f"Cluster {i} Mean Values:")
    print(means)

    # Calculate mode for categorical features
    modes = cluster_data[categorical_features].mode().iloc[0]
    print(f"Cluster {i} Mode Values:")
    print(modes)


df2=pd.read_csv('/Users/raosamvr/Downloads/sdfsdf.csv')
crosstab = pd.crosstab(df['cluster_labels'], df['Breath Sounds'], normalize='index')
for i in range(df['cluster_labels'].nunique()):
    cluster_data = df[df['cluster_labels'] == i]
    proportion = cluster_data['Breath Sounds'].value_counts(normalize=True)
    print(f"Cluster {i} Proportions:")
    print(proportion)

for i in range(df['cluster_labels'].nunique()):
    cluster_data = df[df['cluster_labels'] == i]
    proportion = cluster_data['Smoking'].value_counts(normalize=True)
    print(f"Cluster {i} Proportions:")
    print(proportion)

numerical_features = ['Age', 'FEV1 Value', 'Eosinophils', 
                      'CReactiveProtein', 'PO2', 'Oxygen', 'Oxygen Saturation', 'WBC']
categorical_features = ['Breath Sounds', 'Smoking', 'hypertension', 'heart_failure', 
                        'diabetes_type_two', 'cardiovascular', 'Exacerbations']

num_clusters = df['cluster_labels'].nunique()

cluster_info = pd.DataFrame()


for i in range(num_clusters):
    cluster_data = df[df['cluster_labels'] == i]
    cluster_data_numerical = cluster_data[numerical_features].replace(-1, np.nan)
    means = cluster_data_numerical.mean()
    modes = cluster_data[categorical_features].mode().iloc[0]
    proportions_df = pd.DataFrame()
    for col in categorical_features:
        valid_data = cluster_data[cluster_data[col] != -1]
        col_proportion = valid_data[col].value_counts(normalize=True).to_frame().T
        col_proportion.index = [f'{col}_proportion']
        proportions_df = pd.concat([proportions_df, col_proportion], axis=0)
    cluster_summary = pd.concat([means, modes, proportions_df.stack()], axis=0)
    cluster_summary.name = f'Cluster {i}'
    cluster_info = pd.concat([cluster_info, cluster_summary.to_frame().T], axis=0)

cluster_info.reset_index(drop=True, inplace=True)


output_file = '/Users/raosamvr/Downloads/pca/clusterv.csv'  # Change this to your desired path
cluster_info.to_csv(output_file, index=False)
   
pcadf = pd.read_csv('/Users/raosamvr/Downloads/sdfsdf.csv')


pcadf.fillna(-1, inplace=True)
pcadf.drop(columns=['Gender Identification', 'SerumPH', 'PaCO2', 'COPD vs Asthma'], inplace=True)
pcadf['FEV1 Value']*=1.73
pcadf['Breath Sounds']*=10

categorical_features = ['Exacerbations', 'COPD vs Asthma', 'Breath Sounds', 'hypertension', 'heart_failure', 'diabetes_type_two', 'cardiovascular']
numerical_features = df.columns.difference(categorical_features).tolist()

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(pcadf)

pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

df2['cluster_labels'] = df2['cluster_labels'].replace({0: 'A', 1: 'B', 2: 'C'})
pca_df['Cluster'] = df2['cluster_labels']  

palette = sns.color_palette("bright")

plt.rcParams['font.family'] = 'Arial'
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette=palette, s=100)  # s is size of points
plt.title('PCA of Dataset with Clusters', fontsize=18, fontweight='bold')
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.legend(title='Cluster', loc='best', fontsize='medium', title_fontsize='13')
plt.savefig('/Users/raosamvr/Downloads/pca2_vibrant.png', dpi=600)
plt.show()
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=pcadf.columns)
print(loadings_df)

impact = [max(abs(pc1), abs(pc2)) for pc1, pc2 in zip(loadings_df['PC1'], loadings_df['PC2'])]

top_indices = np.argsort(impact)[-10:]

top_indices = np.argsort(impact)[-8:]

top_features = loadings_df.index[top_indices].tolist()
top_pc1_loadings = loadings_df['PC1'].iloc[top_indices].tolist()
top_pc2_loadings = loadings_df['PC2'].iloc[top_indices].tolist()

x = np.arange(len(top_features))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, top_pc1_loadings, width, label='PC1')
rects2 = ax.bar(x + width/2, top_pc2_loadings, width, label='PC2')

ax.set_ylabel('Loadings')
ax.set_title('PCA Feature Loadings for PC1 and PC2', fontsize = 20, fontweight =  'bold')
ax.set_xticks(x)
ax.set_xticklabels(top_features, rotation=45, fontsize=12, fontweight='bold')
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig('/Users/raosamvr/Downloads/pca1.png', dpi=600)

plt.show()