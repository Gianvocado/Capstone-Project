# Importar los paquetes necesarios
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
import numpy as np
import os


# Se inicializa la tasa de aprendizaje inicial, el número de épocas para entrenar,
# y tamaño del lote.
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Users\gngrc\Desktop\Reconocimiento Mascarilla Mobilenetv2\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Se toma la lista de imágenes en nuestro directorio de conjunto de datos, luego inicializa
# la lista de datos (es decir, imágenes) e imágenes de clase.
print("[INFO] Cargando imagenes...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# Se realiza una codificación one-hot en las etiquetas.
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Se construye el generador de imágenes de entrenamiento para el aumento de datos.
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Carga la red MobileNetV2, asegurándose de que los conjuntos de capas FC principales se dejen
# fuera.
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Se construye la cabeza del modelo que se colocará encima del modelo base.
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Se coloca el modelo FC de la cabeza encima del modelo base (esto se convertirá en
# el modelo real que entrenaremos)
model = Model(inputs=baseModel.input, outputs=headModel)

# Se recorre todas las capas en el modelo base y se congelan para que
# *no* se actualicen durante el primer proceso de formación.
for layer in baseModel.layers:
	layer.trainable = False
    
    
    
    

# Se compila el modelo.
print("[INFO] compilando modelo...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Se entrena la cabeza del modelo.
print("[INFO] entrenando cabeza...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Se hacen predicciones en el conjunto de prueba.
print("[INFO] evaluando red...")
predIdxs = model.predict(testX, batch_size=BS)

# Para cada imagen en el conjunto de prueba, necesitamos encontrar el índice de la etiqueta con la probabilidad
# predicha más grande correspondiente.
predIdxs = np.argmax(predIdxs, axis=1)

# Se muestra un informe de clasificación bien formateado.
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Se serializa el modelo en el disco.
print("[INFO] guardando modelo de detector de mascara...")
model.save("mask_detector.model", save_format="h5")


# Se traza la pérdida de entrenamiento y la precisión
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Entrenando perdida y precision")
plt.xlabel("Epoch #")
plt.ylabel("Perdida/Precision")
plt.legend(loc="lower left")
plt.savefig("plot.png")