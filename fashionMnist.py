# This is a sample Python script.
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from matplotlib import image

# Charger les données fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

import numpy as np

# Press Maj+F10 to execute it or replace it with your code.
# Afficher les dimensions des ensembles d'entraînement et de test
print("Dimensions des ensembles d'entraînement :")
print("Données d'entraînement :", train_images.shape)
print("Étiquettes d'entraînement :", train_labels.shape)

print("\nDimensions des ensembles de test :")
print("Données de test :", test_images.shape)
print("Étiquettes de test :", test_labels.shape)

# Normaliser les données en les mettant à l'échelle entre 0 et 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Ajouter une dimension pour la couleur (canal) car les images MNIST sont en niveaux de gris
#train_images = train_images.reshape((-1, 28, 28, 1))
#test_images = test_images.reshape((-1, 28, 28, 1))

# Convertir les étiquettes en catégories one-hot
#train_labels = to_categorical(train_labels, num_classes=10)
#test_labels = to_categorical(test_labels, num_classes=10)

# Les images sont des tableaux NumPy 28x28, avec des valeurs de pixels allant de 0 à 255.
# Les étiquettes sont un tableau d'entiers, allant de 0 à 9.
# Ceux-ci correspondent à la classe de vêtements que l'image représente :
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

""" PREPARATION VISUEL DES DATA 
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""
# creation du reseau de neurone
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
"""
model = Sequential()

# Ajouter des couches de convolution et de max pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

# Ajouter une couche fully connected avec 64 neurones et une activation ReLU
model.add(Dense(64, activation='relu'))

# Ajouter la couche de sortie avec 10 neurones (correspondant aux 10 classes de MNIST) et une activation softmax
model.add(Dense(10, activation='softmax'))
"""
# COMPILATION ENTRAINEMENT

# Compiler le modèle avec l'optimiseur Adam, la fonction de perte categorical_crossentropy et la métrique accuracy
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Entraîner le modèle avec les données d'entraînement et les étiquettes
model.fit(train_images, train_labels, epochs=10, batch_size=64,
          validation_data=(test_images, test_labels))
# la précision de l'entraînement


# EVALUATION PREDICTION

# Évaluer le modèle sur les données de test
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)  # la précision des tests


# Faire des prédictions sur de nouvelles données

new_data = test_images  # Sélectionne la première image de l'ensemble de test Dans ce cas, [0:1] signifie que nous sélectionnons les données de test de l'indice 0 (la première image) jusqu'à l'indice 1 (l'exclus).

# Assurez-vous que les dimensions de la nouvelle donnée sont correctes
# Les dimensions typiques pour les images MNIST sont (batch_size, 28, 28, 1)
# Si vous avez une seule image, la dimension batch_size doit être 1
print("Dimensions de new_data :", new_data.shape)

# Maintenant, vous pouvez utiliser votre modèle pour faire des prédictions sur cette nouvelle donnée
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Afficher les prédictions
print("Prédictions :", predictions)

print(np.argmax(predictions[0]))



# Afficher l'image de prédiction
# plt.imshow(new_data[0, :, :, 0], cmap='gray')
# plt.axis('off')  # Ne pas afficher les axes
# plt.show()


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
""""
i = 2
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
"""
""""
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
"""


# Grab an image from the test dataset.
#img = test_images[1]
img = image.imread("data2/laine.jpg")

# Normaliser l'image en la mettant à l'échelle entre 0 et 1
img = img / 255.0

#la reformater en dimension 28*28
img = tf.image.resize(img, [28, 28])

# Convertir l'image en niveaux de gris (une seule chaîne de couleur)
img = tf.image.rgb_to_grayscale(img)

# Ajouter une dimension pour la couleur (canal) car votre modèle attend une entrée de forme (28, 28, 1)
img = np.expand_dims(img, axis=0)

# Faire une prédiction sur cette image
predictions_single = probability_model.predict(img)

# Afficher la prédiction
print(predictions_single)

#Afficher l'image et le graphique du % de precision
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(0, predictions_single[0], test_labels, img)
plt.subplot(1,2,2)
plot_value_array(0, predictions_single[0],  test_labels)
plt.show()