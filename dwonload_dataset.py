import pandas as pd

# Option A: Load directly from sklearn (no Kaggle needed)
from sklearn.datasets import fetch_openml

heart = fetch_openml(name='heart-disease', version=1, as_frame=True)
df = heart.frame
df.to_csv('data/heart_disease.csv', index=False)
print("Dataset saved! Shape:", df.shape)
print(df.head())