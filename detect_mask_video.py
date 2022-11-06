# Importar los paquetes necesarios
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# Se toma las dimensiones del marco y luego construye un blob a partir de él.
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# Pasar el blob a través de la red y obtener las detecciones de rostros.
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# Se inicializa nuestra lista de caras, sus ubicaciones correspondientes y la lista de predicciones de
	# nuestra red de máscaras faciales.
	faces = []
	locs = []
	preds = []

	# Bucle sobre las detecciones
	for i in range(0, detections.shape[2]):
		# Se extrae la confianza (es decir, la probabilidad) asociada con la
		# detección.
		confidence = detections[0, 0, i, 2]

		# Se filtran las detecciones débiles asegurándose de que la confianza sea
		# mayor que la confianza mínima.
		if confidence > 0.5:
			# Se calcula las coordenadas (x, y) del cuadro delimitador
			# del objeto.
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Se asegura de que los cuadros delimitadores estén dentro
			# de las dimensiones de el marco.
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            
            
            
            
            

			# Se extrae el ROI de la cara, se convierte de BGR a canal RGB
			# pedido, se cambia el tamaño a 224x224 y se preprocesa.
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Se agrega la cara y los cuadros delimitadores a sus respectivas listas.
			faces.append(face)
			locs.append((startX, startY, endX, endY))
            
            
            

	# Se hacen predicciones si se detectó al menos una cara.
	if len(faces) > 0:
		# Para una inferencia más rápida, se hacen predicciones por lotes
		# en *todas* las caras al mismo tiempo en lugar de predicciones
		# una por una en el bucle `for` anterior.
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# Se devuelve una tupla de 2 de las coincidencias de las caras y sus
	# lugares correspondientes.
	return (locs, preds)

# Se carga el modelo serializado de detector de rostros desde el disco.
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Se carga el modelo de detector de máscara facial desde el disco.
maskNet = load_model("mask_detector.model")

# Se inicializa la transmisión de video
print("[INFO] iniciando transmisión de video...")
vs = VideoStream(src=0).start()

# Bucle sobre los fotogramas de la transmisión de video.
while True:
	# Se toma el marco de la secuencia de video encadenada y cambia su tamaño
	# para que tenga un ancho máximo de 400 píxeles.
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Se detectan rostros en el marco y determinan si llevan puesto una
	# mascarilla o no.
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Se recorren las ubicaciones de rostros detectados y sus
	# ubicaciones correspondientes.
	for (box, pred) in zip(locs, preds):
		# Se descomprime el cuadro delimitador y las predicciones.
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Se determina la etiqueta de clase y el color que usaremos para dibujar
		# el cuadro delimitador y el texto.
		label = "Mascarilla" if mask > withoutMask else "Sin Mascarilla"
		color = (0, 255, 0) if label == "Mascarilla" else (0, 0, 255)

		# Se incluye la probabilidad en la etiqueta.
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Se muestra la etiqueta y el rectángulo del cuadro delimitador
		# en el marco de salida.
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Muestra el cuadro de salida
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Si la tecla `q` fue presionada, se sale del bucle.
	if key == ord("q"):
		break

# Hacer limpieza
cv2.destroyAllWindows()
vs.stop()