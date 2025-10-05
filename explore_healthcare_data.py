import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('healthcare_dataset.csv')

# Basic exploration
print(data.head())
print(data.describe())
print(data['Medical Condition'].value_counts())

# Save a cleaned version (e.g., standardize names)
data['Name'] = data['Name'].str.title()
data.to_csv('cleaned_healthcare_data.csv', index=False)

data['Medical Condition'].value_counts().plot(kind='bar')
plt.title('Distribution of Medical Conditions')
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.show()