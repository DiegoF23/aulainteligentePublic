import cv2
import time

# inicializacion del modelo y etiquetas de clase
prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
aula= "Aula Vacia"
print(aula)
# Weights
model = "model/MobileNetSSD_deploy.caffemodel"
# labels de clases
classes = {0:"background", 1:"aeroplane", 2:"bicycle",
          3:"bird", 4:"bote",
          5:"bottle", 6:"autobus",
          7:"auto", 8:"gato",
          9:"silla", 10:"vaca",
          11:"diningtable", 12:"perro",
          13:"caballo", 14:"motorbike",
          15:"Persona", 16:"pottedplant",
          17:"sheep", 18:"sofa",
          19:"train", 20:"tv/monitor"}
# cargar modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ----------- LECTURA DE CAMARA Y PROCESAMIENTO -----------
cap = cv2.VideoCapture("videoAulaAlumno/aulaVideo.mp4")  # Utiliza la cámara predeterminada (cámara web) 

# Variables para el seguimiento de estado
aula = "Aula Vacia"
person_detected = False
last_detection_time = 0
cooldown_time = 10  # Tiempo de espera (en segundos) antes de ejecutar la siguiente acción

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_resized = cv2.resize(frame, (300, 300))

    # Crear un blob
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

    # ----------- DETECCION Y PREDICCIONES -----------
    net.setInput(blob)
    detections = net.forward()

    current_time = time.time()

    # Verifica si se detecta una persona con confianza//// confianza del 35% de prediccion
    person_detected = any(detection[2] > 0.35 for detection in detections[0][0] if classes[int(detection[1])] == "Persona")

    if person_detected and aula == "Aula Vacia":
        aula = "Aula Ocupada"
        print("Encender Luces")
        last_detection_time = current_time

    if not person_detected and aula == "Aula Ocupada" and (current_time - last_detection_time) >= cooldown_time:
        aula = "Aula Vacia"
        print("Apagar Luces")

    for detection in detections[0][0]:
        # Código para dibujar las detecciones mayor al 75%
        if detection[2] > 0.35:
               if classes[detection[1]] == "Persona" :
                label = classes[detection[1]]
                box = detection[3:7] * [width, height, width, height]
                x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)
                cv2.putText(frame, label, (x_start, y_start - 25), 1, 1.5, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()