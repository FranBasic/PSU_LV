import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

image = mpimg.imread('C:/Users/student/Desktop/LV6/example.png')

if image.max() <= 1.0:
    image = (image * 255).astype(np.uint8)

image_reshaped = image.reshape(-1, 3)

k = 10
kmeans = KMeans(n_clusters=k, n_init=1, random_state=42)
kmeans.fit(image_reshaped)

cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
labels = kmeans.labels_
quantized_image = cluster_centers[labels].reshape(image.shape)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(quantized_image)
plt.title(f'Kvantizirano(k={k})')

plt.show()