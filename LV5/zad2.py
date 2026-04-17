import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv('occupancy_processed.csv')

X = data[['S3_Temp', 'S5_CO2']].values
y = data['Room_Occupancy_Count'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion).plot()
plt.show()

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("\nZAD 2\nTocnost:", accuracy)

print(classification_report(y_test, y_pred))

# e) Model bolje prati manje susjeda, ali veci broj bolje trenira model i male grupe postaju zanemarive
# f) Bez skaliranja ne radi toliko dobro jer udaljenosti nece biti pravilno izracunate