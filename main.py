# This is a sample Python script.
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
import numpy as np

# Press Maj+F10 to execute it or replace it with your code.
# Afficher les dimensions des ensembles d'entraînement et de test
print("Dimensions des ensembles d'entraînement :")
print("Données d'entraînement :", x_train.shape)
print("Étiquettes d'entraînement :", y_train.shape)

print("\nDimensions des ensembles de test :")
print("Données de test :", x_test.shape)
print("Étiquettes de test :", y_test.shape)

# Normaliser les données en les mettant à l'échelle entre 0 et 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ajouter une dimension pour la couleur (canal) car les images MNIST sont en niveaux de gris
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Convertir les étiquettes en catégories one-hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#creation du reseau de neurone
model = Sequential()

# Ajouter des couches de convolution et de max pooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# Press the green button in the gutter to run the script.
model.add(Flatten())
# Ajouter une couche fully connected avec 64 neurones et une activation ReLU
model.add(Dense(64, activation='relu'))

# Ajouter la couche de sortie avec 10 neurones (correspondant aux 10 classes de MNIST) et une activation softmax
model.add(Dense(10, activation='softmax'))


#COMPILATION ENTRAINEMENT

# Compiler le modèle avec l'optimiseur Adam, la fonction de perte categorical_crossentropy et la métrique accuracy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle avec les données d'entraînement et les étiquettes
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))



#EVALUATION PREDICTION

# Évaluer le modèle sur les données de test
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Faire des prédictions sur de nouvelles données

new_data = x_test[0:3]  # Sélectionne la première image de l'ensemble de test Dans ce cas, [0:1] signifie que nous sélectionnons les données de test de l'indice 0 (la première image) jusqu'à l'indice 1 (l'exclus).

# Assurez-vous que les dimensions de la nouvelle donnée sont correctes
# Les dimensions typiques pour les images MNIST sont (batch_size, 28, 28, 1)
# Si vous avez une seule image, la dimension batch_size doit être 1
print("Dimensions de new_data :", new_data.shape)

# Maintenant, vous pouvez utiliser votre modèle pour faire des prédictions sur cette nouvelle donnée
predictions = model.predict(new_data)

# Afficher les prédictions
print("Prédictions :", predictions)


# Afficher l'image de prédiction
plt.imshow(new_data[0, :, :, 0], cmap='gray')
plt.axis('off')  # Ne pas afficher les axes
plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
