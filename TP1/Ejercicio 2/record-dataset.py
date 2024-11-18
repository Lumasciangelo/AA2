import cv2
import os
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Inicializar captura de video (cámara en vivo)
captura = cv2.VideoCapture(0)  # 0 es la cámara por defecto

if not captura.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()


X = []
Y = []
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

        # Mostrar el frame con los puntos clave dibujados
        cv2.imshow("Video en Vivo", frame)
    #etiquetar manualmente las imagenes según si es piedra papel o tijera    
    key = cv2.waitKey(5) & 0xFF
    if key == ord('r') and puntos_mano != []:
        Y.append(0)
        X.append(puntos_mano)
        print(0)
    elif key == ord('p') and puntos_mano != []:
        Y.append(1)
        X.append(puntos_mano)
        print(1)
    elif key == ord('s') and puntos_mano != []:
        Y.append(2)
        X.append(puntos_mano)
        print(2)
    elif key == ord('q'): #Exit with 'q'
        break
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(X)
#print(Y)

# Guardar el array en un archivo .npy
np.save('mi_dataset_X.npy', np.array(X))
np.save('mi_dataset_Y.npy', np.array(Y))

print("Datos guardados!")