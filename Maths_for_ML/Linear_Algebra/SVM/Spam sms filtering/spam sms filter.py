import pandas as pd


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = pd.read_csv(url, sep='/t', names=['label', 'message'], compression='zip')
