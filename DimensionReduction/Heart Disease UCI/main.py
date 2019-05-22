import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
#print(os.listdir())
from sklearn.decomposition import PCA

df = pd.read_csv('heart.csv')
#print(df.head())

corr = df.corr(method='pearson')

labels = {
    'age': 'Age (in years)', # Can be made categorical,
    'sex': 'Sex', # Categorical
    'cp': 'Chest Pain Type', # Categorical
    'trestbps': 'Resting blood pressure',
    'chol': 'Serum Cholestoral',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise Induced Angina', # Categorical,
    'oldpeak': 'ST depression',
    'slope': 'Slope of peak exercise ST Seg',
    'ca': 'No. of major vessels colored by flourospy', # Range (0-3)
    'thal': 'Thal', # Categorical,
    'target': 'Target'
}

corr = corr.rename(labels)

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(240, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0)
# plt.show()
#print(corr)

component_var = {}
for i in range(2, 6):
    pca = PCA(n_components=i)
    res = pca.fit(df)

    # (pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))
    component_var[i] = sum(pca.explained_variance_ratio_)
    print ('Au composant : ', i,' =>', component_var[i])
    # print(component_var[i])