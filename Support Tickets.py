# This model predicts the priority of a support ticket based on various features available at the time of ticket intake.

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier

# Load and clean data
data = pd.read_csv("C://Users//akjee//Documents//ML//Support_tickets.csv")
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
print("Data Size:", data.shape)
print(data.head(10))

# Convert object columns to 'category' dtype for LightGBM
'''1. data.select_dtypes(include=['object'])
This selects all columns in data that have the object data type.

These are typically string columns like names, categories, labels, etc.

2. .columns
Extracts the column names of those object-type columns.

3. data[...]
Uses those column names to subset the original DataFrame.

This is the part that says: “I want to modify these specific columns.”

4. .astype('category')
Converts the selected columns from object to category.

5. data[...] = ...
Assigns the converted columns back to the original DataFrame, replacing the old object columns.'''

data[data.select_dtypes(include=['object']).columns] = data.select_dtypes(include=['object']).astype('category')

# Split features and target
X = data.iloc[:, :-1]  # Independent variables
y = data.iloc[:, -1]   # Dependent variable

# Identify categorical column indices for LightGBM
cat_features = list(X.select_dtypes(include='category').columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(f"X Train shape is :{X_train.shape}")
print(f"X Test shape is :{X_test.shape}")
print(f"Y Train shape is :{y_train.shape}")
print(f"Y Test shape is :{y_test.shape}")

# Train classifier
classifier = LGBMClassifier(random_state=42, verbose=-1)
classifier.fit(X_train, y_train, categorical_feature=cat_features)

# Predictions
y_pred = classifier.predict(X_test)
print("Predictions:", y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report and accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model Accuracy:", classifier.score(X_test, y_test))