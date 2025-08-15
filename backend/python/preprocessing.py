import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script

data_path = os.path.join(base_path, "../data/pima.csv")
data = pd.read_csv(data_path)

# checking functions - multi use
def round(data):
    round=(data.isin([0, None, np.nan]).sum() / data.shape[0] * 100).round(2)
    return round
    
# IQR Method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    return df[(df[column] >= lb) & (df[column] <= ub)]

# Z-score Method - not used as the data has lots of outliers 
def remove_outliers_zscore(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    lb = mean - threshold * std
    ub = mean + threshold * std
    return df[(df[column] >= lb) & (df[column] <= ub)]

print(data.head())
data.info()
print(data.isnull().sum())

## STATISTICAL ANALYSIS
print(data.describe())

## CHECK CATEGORICAL AND NUMERICAL COLUMNS
categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
print("Categorical Columns: ", categorical_columns)
numerical_columns = [col for col in data.columns if data[col].dtype != 'object']
print("Numerical Columns: ", numerical_columns)


## CHECKING FOR MISSING DATA
round(data)

## HANDLING THE MISSING DATA
# dropping glucose bp and bmi
data = data.drop(data[data['Glucose'] == 0].index)
data = data.drop(data[data['BloodPressure'] == 0].index)
data = data.drop(data[data['BMI'] == 0].index)
round(data)
# median of skin thickness, pregnancies and insulin 
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].median())
data['Pregnancies'] = data['Pregnancies'].replace(0, data['Pregnancies'].median())
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].median())
round(data)

## REMOVING OUTLIERS
# Check outliers for pregnancies, BMI, insulin, blood pressure (before preprocessing)
fig, axs = plt.subplots(4, 1, dpi=95, figsize=(7, 17))
fig.suptitle("Outliers Before Preprocessing", fontsize=16)  # Add heading
i = 0
for col in ['Pregnancies', 'BMI', 'Insulin', 'BloodPressure']:
    axs[i].boxplot(data[col], vert=False)
    axs[i].set_ylabel(col)
    i += 1
plt.show()

# Removing outliers (IQR method chosen as it's more robust)
data = remove_outliers_iqr(data, 'BMI')
data = remove_outliers_iqr(data, 'Insulin')
data = remove_outliers_iqr(data, 'BloodPressure')
data = remove_outliers_iqr(data, 'Pregnancies')

# Checking outliers after removing
fig, axs = plt.subplots(4, 1, dpi=95, figsize=(7, 17))
fig.suptitle("Outliers After Preprocessing", fontsize=16)  # Add heading
i = 0
for col in ['Pregnancies', 'BMI', 'Insulin', 'BloodPressure']:
    axs[i].boxplot(data[col], vert=False)
    axs[i].set_ylabel(col)
    i += 1
plt.show()

## EDA: Distribution of Numerical Features
fig, axs = plt.subplots(4, 2, dpi=95, figsize=(14, 16))
fig.suptitle("Distribution of Numerical Features", fontsize=16)
numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for i, col in enumerate(numerical_columns):
    sns.histplot(data[col], kde=True, ax=axs[i // 2, i % 2], color='blue')
    axs[i // 2, i % 2].set_title(f"Distribution of {col}")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

## EDA: Correlation Heatmap
plt.figure(dpi=95, figsize=(10, 8))
plt.title("Correlation Heatmap", fontsize=16)
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.show()

## EDA: Outcome Distribution
plt.figure(dpi=95, figsize=(6, 4))
sns.countplot(x='Outcome', data=data, palette='Set2')
plt.title("Outcome Distribution", fontsize=16)
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()

print(data.duplicated().sum()) # duplicate checking

# Save as Pickle (for faster loading)
pickle_path = os.path.join(base_path, "../data/preprocessed_pima.pkl")
with open(pickle_path, "wb") as f:
    pickle.dump(data, f)
print("Data saved successfully as Pickle.")

# Save as CSV
csv_path = os.path.join(base_path, "../data/preprocessed_pima.csv")
data.to_csv(csv_path, index=False)
print("Data saved successfully as CSV.")

# Balancing the dataset using undersampling
class_0 = data[data['Outcome'] == 0]
class_1 = data[data['Outcome'] == 1]

# Undersample the majority class
class_0_undersampled = class_0.sample(len(class_1), random_state=42)
data_balanced = pd.concat([class_0_undersampled, class_1])

# Shuffle the balanced dataset
data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset
balanced_csv_path = os.path.join(base_path, "../data/balanced_pima.csv")
data_balanced.to_csv(balanced_csv_path, index=False)
print("Balanced dataset saved successfully as CSV.")
