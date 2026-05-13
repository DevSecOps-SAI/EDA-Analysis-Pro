import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Set styling for plots
sns.set_theme(style="whitegrid")
%matplotlib inline 

# 1. LOAD DATA
# Replace 'data.csv' with your dataset path
df = pd.read_csv('your_dataset.csv')

# 2. INITIAL DATA INSPECTION
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# 3. DATA CLEANING
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Handle missing values (Example: fill with mean or drop)
# df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# 4. UNIVARIATE ANALYSIS (One Variable at a time)
# Distribution of a numerical column
plt.figure(figsize=(10, 5))
sns.histplot(df['numeric_column'], kde=True, color='blue')
plt.title('Distribution of Numeric Column')
plt.show()

# Count of a categorical column
plt.figure(figsize=(10, 5))
sns.countplot(x='category_column', data=df)
plt.title('Count of Categories')
plt.xticks(rotation=45)
plt.show()

# 5. BIVARIATE ANALYSIS (Relationship between two variables)
# Scatter plot for two numerical columns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='column_x', y='column_y', data=df, hue='category_column')
plt.title('Column X vs Column Y')
plt.show()

# Box plot to see distribution across categories
plt.figure(figsize=(10, 6))
sns.boxplot(x='category_column', y='numeric_column', data=df)
plt.title('Numeric Value by Category')
plt.show()

# 6. MULTIVARIATE ANALYSIS
# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
