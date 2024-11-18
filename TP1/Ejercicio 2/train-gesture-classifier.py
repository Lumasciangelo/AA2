import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Cargar los datos desde un archivo .npy
X = np.load('mi_dataset_X.npy')
Y = np.load('mi_dataset_Y.npy')

# Crear un modelo simple (puedes personalizar este modelo)
model = models.Sequential([
    layers.Input(shape=(21,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Guardar el modelo en un archivo .h5
model.save('mi_modelo.h5')

print("Modelo guardado en 'mi_modelo.h5'.")
