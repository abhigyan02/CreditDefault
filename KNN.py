import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

creditData = pd.read_csv('credit_data.csv')

features = creditData[['income', 'age', 'loan']]
target = creditData.default

X = np.array(features).reshape(-1, 3)
Y = np.array(target)

X = preprocessing.MinMaxScaler().fit_transform(X)

features_train, features_test, target_train, target_test = train_test_split(X, Y, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=32)
model = model.fit(features_train, target_train)
predictions = model.predict(features_test)

cross_valid_score = []
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    cross_valid_score.append(scores.mean())

print("Optimal K with cross-validation: ", np.argmax(cross_valid_score))

confusionMatrix = confusion_matrix(target_test, predictions, labels=model.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=model.classes_)
display.plot()
plt.show()

print(accuracy_score(target_test, predictions))
