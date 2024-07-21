import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/home/vikrant/5th SEm/EDA/Project/DEV/resampled_try1.csv')  # Replace 'your_dataset.csv' with your dataset file

# Select the column containing numerical categories
numerical_column = 'dx'  # Replace 'numerical_category_column_name' with your column name

# Count the occurrences of each numerical category
category_counts = df[numerical_column].value_counts().sort_index()

# Plot the counts
plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar', color='skyblue')
plt.title('Counts of Numerical Categories')
plt.xlabel('Numerical Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
