from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

print(iris_df.head())

X = iris_df.drop('species', axis=1)  
y = iris_df['species']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
