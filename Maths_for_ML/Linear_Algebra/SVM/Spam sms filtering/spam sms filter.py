import pandas as pd
from sklearn.preprocessing import LabelEncoder


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = pd.read_csv(url, sep='/t', names=['label', 'message'], compression='zip')

#encoding the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
