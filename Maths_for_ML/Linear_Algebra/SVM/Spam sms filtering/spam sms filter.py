import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score



df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

#encoding the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])


# Split the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=42)

#convert the text to numerical value

vector = CountVectorizer()

x_train = vector.fit_transform(x_train)
x_test = vector.transform(x_test)

#time to train the data

classifier = SVC(kernel='linear', gamma='scale')
classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(x_test)

#Evaluation

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy score:{accuracy*100:.2f}%")

#Classification Report 
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))





