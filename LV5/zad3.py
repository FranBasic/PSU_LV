import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay

data = pd.read_csv('occupancy_processed.csv')

X = data[["S3_Temp", "S5_CO2"]].values
y = data["Room_Occupancy_Count"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

dt = DecisionTreeClassifier(max_depth=2, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=confusion).plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("\nZAD 3\nTocnost:", accuracy)

print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=["S3_Temp", "S5_CO2"], class_names=True)
plt.show()

# b) Povecanjem max-deptha povecavamo tocnost treniranja
# c) Rezultati se ne mijenjaju toliko u slucaju ne koristenja skaliranja