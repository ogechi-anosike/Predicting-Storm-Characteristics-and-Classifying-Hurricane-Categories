import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

data = pd.read_csv('storms.csv')
print(data.head())

# replace null values with median value of the column 
data['tropicalstorm_force_diameter'].fillna(data['tropicalstorm_force_diameter'].median(), inplace=True)
data['hurricane_force_diameter'].fillna(data['hurricane_force_diameter'].median(), inplace=True)
#replace null values with zero 
data['category'].fillna(0, inplace=True)

label_encoder = LabelEncoder()
data['status'] = label_encoder.fit_transform(data['status'])  # Encoding the 'status' column
data['category'] = label_encoder.fit_transform(data['category'])  


X = data.drop(['category', 'name'], axis=1)  
y = data['category']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# nodel training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# model evaluation 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5])

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()





