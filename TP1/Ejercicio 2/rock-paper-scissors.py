import tensorflow as tf
import cv2
import os
import mediapipe as mp
import numpy as np


# Cargar el modelo desde el archivo .h5
model = tf.keras.models.load_model('mi_modelo.h5')

# Mostrar un resumen del modelo para verificar que se ha cargado correctamente
#model.summary()

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar captura de video (cámara en vivo)
captura = cv2.VideoCapture(0)  # 0 es la cámara por defecto

if not captura.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

# Bucle para capturar video en vivo
while True:
    # Leer un frame
    ret, frame = captura.read()
    
    # Si no se puede leer el frame, salir del bucle
    if not ret:
        print("No se puede obtener el frame.")
        break
    
    # Convertir BGR a RGB (MediaPipe trabaja con imágenes en RGB)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Lista para almacenar las coordenadas de los 21 puntos
    puntos_mano = []

    # Inicializar la detección de manos
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(image_rgb)

        # Si se detectan manos, dibujar los puntos clave en la imagen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    alto, ancho, _ = frame.shape
                    # Convertir las coordenadas relativas a píxeles
                    x, y = int(lm.x * ancho), int(lm.y * alto)
                    puntos_mano.append((x, y))
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)



    if puntos_mano != []:
        # Realizar la predicción
        predictions = model.predict(np.array([puntos_mano]))

        # Obtener la etiqueta predicha (índice de la clase con la mayor probabilidad)
        predicted_label = np.argmax(predictions, axis=1) 

        if predicted_label == 0:
            texto = 'piedra'
        elif predicted_label == 1:
            texto = 'papel'
        elif predicted_label == 2:
            texto = 'tijera'
    else:
        texto = 'no se detectaron manos'

    cv2.putText(frame, texto, 
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Mostrar el frame con los puntos clave dibujados
    cv2.imshow("Video en Vivo", frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): #Exit with 'q'
        break

cv2.waitKey(0)
cv2.destroyAllWindows()