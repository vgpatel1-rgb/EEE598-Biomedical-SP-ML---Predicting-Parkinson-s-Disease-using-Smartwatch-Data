# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:28:42 2025

@author: jcmir
"""
# Load all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats as st
import statsmodels.stats.multitest as smm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#%% Statistical Analysis

# Calculate Cliff's delta function to calculate Cliff's delta for magnitude of 
# response
def calculate_cliff_delta(x,y):
    n1 = len(x)
    n2 = len(y)
    greater = 0
    less = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                greater = greater +1
            elif xi < yi:
                less = less +1
    delta = (greater - less) / (n1 * n2)
    return delta

# Import features and labels from pre-processing step
feature_df = pd.read_csv("all_stratified_subjects_features.csv",header=0)
labels_df = pd.read_csv("modified_stratified_patient_data.csv",header=0)

# Extract features and labels from dataframes
X = feature_df.drop(columns=['subject_id'])
Y = labels_df[['label','id']]

# Split into 80-20 train-test stratifying on labels 
# Index is based on order in original csv files
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
                                                    random_state=5, stratify=Y['label'])


# Make full train set
train_set = pd.concat([X_train,Y_train],axis=1)

# Three possible labels
label_types = [0,1,2]
# Initialize for for loop
kruskal_results = []
# Kruskal Wallis Test across three groups saving statistic and pvalue
for feature in X_train.columns:
    data = [train_set[train_set['label'] == label][feature].values 
            for label in label_types]
    statistic, pvalue = st.kruskal(*data)
    kruskal_results.append({'Feature':feature,'Statistic':statistic,'P-Value':pvalue})
# Save to data frame
kruskal_results_df = pd.DataFrame(kruskal_results)

# Drop non-significant features (pvalue > 0.05)
significant_results = kruskal_results_df[kruskal_results_df['P-Value']<0.05]
print(f"{len(significant_results)} out of {len(kruskal_results_df)} features are significant")

# Run Post Hoc with Mann-Whitney
# Define three comparisons that are going to occur
comparisons = [(1,0,'PD vs HC'),(1,2,'PD vs DD'),(0,2,'HC vs DD')]
# Initalize for for loop
mann_results = []
# Go through each comparison for each feature and run Mann-Whitney U Test
for feature in significant_results['Feature']:
    for group1, group2, name in comparisons:
        data1 = train_set[train_set['label']==group1][feature].values
        data2 = train_set[train_set['label']==group2][feature].values
        statistic, pvalue = st.mannwhitneyu(data1, data2, alternative = 'two-sided')
        delta = calculate_cliff_delta(data1, data2)
        mann_results.append({'Feature':feature,'Comparison':name,
                        'Statistic':statistic,'P-Value':pvalue,'Delta':delta})
# Save as dataframe
mann_results_df = pd.DataFrame(mann_results)
# Run Benjamini-Hochman multiple hypothesis correction and add to dataframe
mann_results_df['FDR'] = smm.multipletests(mann_results_df['P-Value'],method='fdr_bh')[1]
mann_results_df=mann_results_df.sort_values('FDR')
# Keep only significant comparisons (pvalue < 0.05, cliff's delta > 0.3)
filtered_mann_results = mann_results_df[(mann_results_df['P-Value'] < 0.05) &
                                        (mann_results_df['Delta'].abs() >= 0.3)]
# Only need to differentiate between PD and healthy and PD and DD
needed_comparison = ['PD vs HC','PD vs DD']
final_features = filtered_mann_results[filtered_mann_results['Comparison'].isin(needed_comparison)]['Feature'].unique()

print(f"{len(final_features)} out of {len(kruskal_results_df)} features are significant and critical")
#%% Unsupervised Learning

# Keep working with only significant and critical features
X_final = X_train[final_features]
# Split train set into test and train for determining unsupervised learning
# effectiveness
X_final_train, X_final_test, Y_final_train, Y_final_test = train_test_split(
    X_final, Y_train, test_size = 0.2, random_state = 5, stratify=Y_train['label'])
# Scale data based on train
scaler = StandardScaler()
X_final_train_scale = scaler.fit_transform(X_final_train)
X_final_test_scale = scaler.transform(X_final_test)
# Run PCA fit to train
pca = PCA(random_state=5)
X_final_train_pca = pca.fit_transform(X_final_train_scale)
variances = np.cumsum(pca.explained_variance_ratio_)
components = range(1, len(variances)+1)
# Plot cumulative explained variance with lines for 90%, 95%, and 99%
plt.figure(figsize=(10,8),layout='constrained')
plt.plot(components,variances,marker='o')
plt.title('PCA Explained Variance Versus Number of Components')
plt.xticks(np.arange(0, 150, 10.0))
plt.axhline(y=0.9, color='r',linestyle='--')
plt.axhline(y=0.95, color='r',linestyle='--')
plt.axhline(y=0.99, color='r',linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
# Print number of components needed for 90%, 95%, and 99% variance
n_components = np.argmax(variances >= 0.9) + 1
print(f"{n_components} PCA components needed to explain 90% of variance")
n_components = np.argmax(variances >= 0.95) + 1
print(f"{n_components} PCA components needed to explain 95% of variance")
n_components = np.argmax(variances >= 0.99) + 1
print(f"{n_components} PCA components needed to explain 99% of variance")
# Run a PCA with 2 components for plotting
pca_2 = PCA(n_components=2, random_state=5)
pca_2_train = pca_2.fit(X_final_train_scale)
X_train_pca_2 = pca_2_train.transform(X_final_train_scale)
X_test_scaled = scaler.transform(X_final_test)
X_test_pca_2 = pca_2_train.transform(X_test_scaled)
# Make train DataFrame and test DataFrame
pca_train_df =pd.DataFrame(data= X_train_pca_2,columns=['PC1','PC2'])
pca_train_df['Label'] = Y_final_train['label'].values
pca_test_df=pd.DataFrame(data= X_test_pca_2,columns=['PC1','PC2'])
pca_test_df['Label'] = Y_final_test['label'].values
# Plot two scatters plot to see train vs test for first two components
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(12,6)
sns.scatterplot(ax=ax1,x='PC1',y='PC2',hue='Label',
                palette={0:'mediumseagreen',1:'red',2:'orange'},data=pca_train_df)
sns.scatterplot(ax=ax2,x='PC1',y='PC2',hue='Label',
                palette={0:'mediumseagreen',1:'red',2:'orange'},data=pca_test_df)
ax1.set_title('Train Data')
ax2.set_title('Test Data')
ax1.legend(labels=['DD','HC','PD'])
ax2.legend(labels=['DD','HC','PD'])
ax1.grid(True,alpha=0.3)
ax2.grid(True,alpha=0.3)
fig.suptitle('PC1 vs PC2 for Train and Test Sets')
fig.tight_layout()
plt.show()

# Make dataframes of full overall train set
all_features_scaled = np.vstack([X_final_train_scale, X_final_test_scale])
labels = np.concatenate([Y_final_train['label'].values,Y_final_test['label'].values])
# Apply t-SNE with 2 components for plotting, 
tsne = TSNE(n_components=2, random_state=30, init='pca', learning_rate='auto')
features_tsne = tsne.fit_transform(all_features_scaled)
tsne_df = pd.DataFrame(data = features_tsne, columns = ['t-SNE 1', 't-SNE 2'])
tsne_df['Label'] = labels
# Plot t-SNE components
plt.figure(figsize=(10, 7))
sns.scatterplot(x="t-SNE 1", y="t-SNE 2", hue="Label", data=tsne_df, 
                palette={0:'mediumseagreen',1:'red',2:'orange'},alpha=0.6,s=15)
plt.title('t-SNE Projection of PADS Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(labels=['DD','HC','PD'])
plt.grid(True, alpha=0.3)
plt.show()

#%% Save all features needed for supervised learning section

# Save full set split 
# Train
buffer_save = pd.concat([Y_train,X_train],axis=1)
buffer_save.to_csv('train_full_features.csv',index=False)
# Test
buffer_save = pd.concat([Y_test,X_test],axis=1)
buffer_save.to_csv('test_full_features.csv',index=False)

# Save significant and critical features
# Train
buffer_save = pd.concat([Y_train, X_final],axis=1)
buffer_save.to_csv('train_significant_critical_features.csv',index=False)
# Test
X_test_final = X_test[final_features]
buffer_save = pd.concat([Y_test,X_test_final],axis=1)
buffer_save.to_csv('test_significant_critical_features.csv',index=False)

Y_train_no_index = pd.DataFrame(Y_train.values)
Y_train_no_index.columns = ['label','id']
Y_test_no_index = pd.DataFrame(Y_test.values)
Y_test_no_index.columns = ['label','id']

# Save 21 PCA componets for 90% of variance
X_train_scaled = scaler.fit_transform(X_final)
X_test_scaled = scaler.transform(X_test_final)
pca_21 = PCA(n_components=21,random_state=5)
X_train_pca21 = pca_21.fit_transform(X_train_scaled)
X_train_pca21 = pd.DataFrame(X_train_pca21)
X_test_pca21 = pca_21.transform(X_test_scaled)
X_test_pca21 = pd.DataFrame(X_test_pca21)
# Train
buffer_save = pd.concat([Y_train_no_index,X_train_pca21],axis=1)
buffer_save.to_csv('train_pca_90.csv',index=False)
# Test
buffer_save = pd.concat([Y_test_no_index,X_test_pca21],axis=1)
buffer_save.to_csv('test_pca_90.csv',index=False)

# Save 31 PCA components for 95% of variance
pca_31 = PCA(n_components=31,random_state=5)
X_train_pca31 = pca_31.fit_transform(X_train_scaled)
X_train_pca31 = pd.DataFrame(X_train_pca31)
X_test_pca31 = pca_31.transform(X_test_scaled)
X_test_pca31 = pd.DataFrame(X_test_pca31)
# Train
buffer_save = pd.concat([Y_train_no_index,X_train_pca31],axis=1)
buffer_save.to_csv('train_pca_95.csv',index=False)
# Test
buffer_save = pd.concat([Y_test_no_index,X_test_pca31],axis=1)
buffer_save.to_csv('test_pca_95.csv',index=False)

# Save 53 PCA components for 99% of variance
pca_53 = PCA(n_components=53,random_state=5)
X_train_pca53 = pca_53.fit_transform(X_train_scaled)
X_train_pca53 = pd.DataFrame(X_train_pca53)
X_test_pca53 = pca_53.transform(X_test_scaled)
X_test_pca53 = pd.DataFrame(X_test_pca53)
# Train
buffer_save = pd.concat([Y_train_no_index,X_train_pca53],axis=1)
buffer_save.to_csv('train_pca_99.csv',index=False)
# Test
buffer_save = pd.concat([Y_test_no_index,X_test_pca53],axis=1)
buffer_save.to_csv('test_pca_99.csv',index=False)