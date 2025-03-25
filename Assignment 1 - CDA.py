#Assignment 1 - CDA

# Import necessary libraries
import pandas as pd  # For handling data (loading, cleaning, and manipulation)
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
import seaborn as sns  # For advanced visualization

# Load training data (contains X and Y)
df_train = pd.read_csv(r"C:\Users\snehi\Documents\CDA_Assignments\case1Data.csv")

# Load new data (contains only X, without Y)
df_new = pd.read_csv(r"C:\Users\snehi\Documents\CDA_Assignments\case1Data_Xnew.csv")

# Display first few rows of training data
print(df_train.head())

# Extract response variable y
Y_train = df_train['y']

# Extract feature matrix X
X_train = df_train.drop(columns=['y'])

# Display dimensions
print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", Y_train.shape)


print(df_train.shape)

print(df_train.columns)

# Count missing values in each column
print(df_train.isnull().sum())

# Check data types
print(df_train.dtypes)

# Plot histogram of Y
plt.figure(figsize=(8,5))
sns.histplot(df_train['Y'], bins=20, kde=True)
plt.title('Distribution of Y')
plt.xlabel('Y')
plt.ylabel('Frequency')
plt.show()