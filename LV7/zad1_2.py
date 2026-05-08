import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.axis('off')
plt.suptitle("Primjeri iz train skupa")
plt.show()

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = x_train_s.reshape((x_train.shape[0], 784))
x_test_s = x_test_s.reshape((x_test.shape[0], 784))

y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

fitt = model.fit(x_train_s, y_train_s, epochs=5, batch_size=32)

train_loss, train_accuracy = model.evaluate(x_train_s, y_train_s)
test_loss, test_accuracy = model.evaluate(x_test_s, y_test_s)

print(f"Tocnost na train skupu: {train_accuracy:.4f}")
print(f"Tocnost na test skupu: {test_accuracy:.4f}")

classes = model.predict(x_test_s)
y_pred_classes = np.argmax(classes, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(cmap=plt.cm.Blues)
plt.title("Matrica zabune na test skupu")
plt.show()

wrong_predictions = np.where(y_test != y_pred_classes)[0]

plt.figure(figsize=(10, 5))
for i in range(5):
    idx = wrong_predictions[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx], cmap=plt.get_cmap('gray'))
    plt.title(f"Stvarna: {y_test[idx]}\nPred: {y_pred_classes[idx]}")
    plt.axis('off')
plt.suptitle("Pogresne klasifikacije")
plt.show()