import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('occupancy_processed.csv')

X = df[['S3_Temp', 'S5_CO2']].values
y = df['Room_Occupancy_Count'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#skaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=confusion).plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("\nZAD 4\nTocnost:", accuracy)

print(classification_report(y_test, y_pred))

# a) Tocnost je manja nego kod KNN modela, ali mrvu bolja nego kod stabla odlucivanja
# b) Uzrok ovim rezultatima je to da model preferira jednu klasu preko druge