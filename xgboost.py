# import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the Dataset 
dataset = pd.read_csv(r'C:\Users\Admin\Desktop\Churn_Modelling.csv')
x = dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

# one hot Encoding the 'Geography' column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(random_state=0) 
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy 
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

# bias score
bias = classifier.score(x_train,y_train)
bias


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 8)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))


# confusion matrix visualization 
from sklearn.metrics import ConfusionMatrixDisplay
plt.figure(figsize=(6,4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Churn', 'Churn'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - XGBoost")
plt.show()


# actual & predicted visualizations 
plt.figure(figsize=(8,4))
plt.plot(y_test[:50], label='Actual', marker='o')
plt.plot(y_pred[:50], label='Predicted', marker='x')
plt.title("Actual vs Predicted Churn (First 50 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Churn (0 = No, 1 = Yes)")
plt.legend()
plt.grid(True)
plt.show()

# training & testing accuracy visualizations 
scores = [bias, ac]
labels = ['Training Accuracy (Bias)', 'Test Accuracy']
plt.figure(figsize=(6,4))
plt.bar(labels, scores)
plt.ylim(0,1)
plt.title("Bias vs Accuracy")
plt.ylabel("Score")
plt.show()








