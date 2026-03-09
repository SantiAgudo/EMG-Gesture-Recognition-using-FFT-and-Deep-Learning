import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ruta a la carpeta de test
test_dir = '/content/drive/My Drive/PAPER/Test'

# Cargar el modelo entrenado
model = tf.keras.models.load_model('/content/drive/My Drive/PAPER/modelo_movimientos.h5')

# Obtener una lista de todas las carpetas de movimientos en la carpeta de prueba
movements = os.listdir(test_dir)
movement_labels = {movement: index for index, movement in enumerate(movements)}

# Inicializar listas para etiquetas verdaderas y predichas
y_true = []
y_pred = []

# Iterar sobre cada carpeta de movimiento
for movement in movements:
    movement_path = os.path.join(test_dir, movement)

    # Iterar sobre cada imagen en la carpeta de movimiento
    for image_name in os.listdir(movement_path):
        image_path = os.path.join(movement_path, image_name)

        # Cargar y preprocesar la imagen
        img = image.load_img(image_path, target_size=(250, 250))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Escalar los valores de la imagen

        # Hacer una predicción
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])

        # Agregar las etiquetas verdadera y predicha a las listas
        y_true.append(movement_labels[movement])
        y_pred.append(predicted_class_index)

# Convertir y_true y y_pred a arrays de numpy para mayor compatibilidad
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Crear la matriz de confusión original
cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12])



# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3,4,5,6,7,8,9,10,11,12])
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.savefig('matriz_confusion.png')  # Guardar la matriz de confusión como una imagen
plt.show()
