import idx2numpy
import numpy as np
import random
import matplotlib.pyplot as plt

images_path = "processed_test/idx/train-images.idx3-ubyte"
labels_path = "processed_test/idx/train-labels.idx1-ubyte"

images = idx2numpy.convert_from_file(images_path)
labels = idx2numpy.convert_from_file(labels_path)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Unique labels:", np.unique(labels))

for i in range(5):
    idx = random.randint(0, len(images)-1)
    img = images[idx]
    label = labels[idx]

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()