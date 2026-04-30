import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

imageNew = mpimg.imread('C:/Users/student/Desktop/LV6/example_grayscale.png')

if len(imageNew.shape) == 2:
    imageNew = np.stack([imageNew] * 3, axis=-1)

if imageNew.dtype != np.uint8:
    imageNew = (imageNew * 255).astype(np.uint8)

image_reshaped = imageNew.reshape((-1, 3))

k = 10
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(image_reshaped)

cluster_centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
quantized_image = cluster_centers[kmeans.labels_].reshape(imageNew.shape)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(imageNew)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(quantized_image)
plt.title(f"Kvantizirano(k={k})")
plt.axis("off")

plt.show()

original_size = imageNew.size * imageNew.itemsize
centers_size = k * 3 * imageNew.itemsize

if k <= 256:
    label_dtype = np.uint8
else:
    label_dtype = np.uint16

labels_size = image_reshaped.shape[0] * np.dtype(label_dtype).itemsize
quantized_size = centers_size + labels_size
compression_ratio = original_size / quantized_size

print(f"Original: {original_size} bytes")
print(f"Kvantizirana slika: {quantized_size} bytes")
print(f"Kompresirano sa {k} klastera: {compression_ratio:.2f}x")

#1. Povećanjem broja klastera dobivamo detaljniju sliku, oštri prijelazi (poput krzna) će izgledati sličnije originalu i sa
#   manjim brojem klastera dok lagani prijelazi (poput neba) zahtjevaju puno više klustera kako bi bili slični originalu  
#2. Sa 10 klastera sliku možemo kompresirati sa 3685400 na 1228806 byteova (3 puta manje byteova) 