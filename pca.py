from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
R = np.array(iris.data)

R_cov = np.cov(R, rowvar=False)

import pandas as pd
iris_covmat = pd.DataFrame(data=R_cov, columns=iris.feature_names)
iris_covmat.index = iris.feature_names
iris_covmat

eig_values, eig_vectors = np.linalg.eig(R_cov)

eig_values
eig_vectors

featureVector = eig_vectors[:,:2]
featureVector

featureVector_t = np.transpose(featureVector)

R_t = np.transpose(R)

newDataset_t = np.matmul(featureVector_t, R_t)
newDataset = np.transpose(newDataset_t)

newDataset.shape

import seaborn as sns
import pandas as pd

df = pd.DataFrame(data=newDataset, columns=['PC1', 'PC2'])
y = pd.Series(iris.target)
y = y.replace(0, 'setosa')
y = y.replace(1, 'versicolor')
y = y.replace(2, 'virginica')
df['Target'] = y 

sns.lmplot(x='PC1', y='PC2', data=df, hue='Target', fit_reg=False, legend=True)