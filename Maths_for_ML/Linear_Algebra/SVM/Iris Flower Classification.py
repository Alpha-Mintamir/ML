from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

print(iris_df.head())

#print

X = iris_df.drop('species', axis=1)  
y = iris_df['species']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svm_model = SVC(kernel='linear', random_state=42)  
svm_model.fit(X_train_scaled, y_train)  


y_pred = svm_model.predict(X_test_scaled)  
accuracy = accuracy_score(y_test, y_pred) 

#print
print(f"Accuracy of SVM on test set: {accuracy:.2f}")



