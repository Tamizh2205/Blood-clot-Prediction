#%%
# new file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ──────────────────────────────────────────
df = pd.read_csv('data/heart_disease.csv')

print(df.head())
print(df['target'].value_counts())

print("=== Dataset Info ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nClass balance:\n{df['target'].value_counts()}")

# ── Plot 1: Class distribution ─────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df,
              palette=['#5DCAA5', '#D85A30'])
plt.title('Class Distribution (0=No Clot Risk, 1=Clot Risk)')
plt.xticks([0, 1], ['No Risk', 'High Risk'])
plt.tight_layout()
plt.savefig('plots/class_distribution.png')
plt.show()

# ── Plot 2: Feature correlations ───────────────────────
df_numeric = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.show()

# ── Plot 3: Age distribution by risk ──────────────────
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', hue='target',
             kde=True, palette=['#5DCAA5', '#D85A30'],
             bins=30)
plt.title('Age Distribution by Clot Risk')
plt.tight_layout()
plt.savefig('plots/age_distribution.png')
plt.show()

# ── Plot 4: Key risk factors boxplots ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ['age', 'trestbps', 'chol', 'thalach']
labels   = ['Age', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate']

for ax, feat, label in zip(axes.flatten(), features, labels):
    sns.boxplot(x='target', y=feat, data=df,
                palette=['#5DCAA5', '#D85A30'], ax=ax)
    ax.set_title(f'{label} by Risk Level')
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig('plots/risk_factor_boxplots.png')
plt.show()

print("\nEDA complete. Plots saved to /plots/")