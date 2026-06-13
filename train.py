import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Rutas de los directorios
dataset_dir = r"C:\Users\aldom\Desktop\Entrenamiento_ia_arenia\Dataset_SinFondo"
model_save_path = "model_unquant.tflite"
labels_save_path = "labels.txt"

# 1. Mapeo estricto de nombres requerido
class_mapping = {
    "CORMORANES NEOTROPICALES": "cormoran_neotropical",
    "GARZA AZUL": "garza_azul",
    "GAVIOTA PERUANA": "gaviota_peruana",
    "PELICANO PERUANO": "pelicano_peruano"
}

# Parámetros
img_height, img_width = 224, 224
batch_size = 16
epochs = 50

# 2. Data Augmentation
# Aplicamos rotación, zoom, y flip horizontal
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 # 20% para validación
)

print("Cargando imágenes de entrenamiento...")
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

print("Cargando imágenes de validación...")
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

validation_generator = validation_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 3. Transfer Learning con MobileNetV2
# Cargamos el modelo base pre-entrenado en ImageNet sin la capa superior
print("Cargando modelo base MobileNetV2...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Congelamos la base para no modificar los pesos durante el entrenamiento inicial
base_model.trainable = False

# Agregamos nuestras capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) # Ayuda a prevenir el overfitting
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Construimos el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilamos el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Entrenamiento
print("Iniciando entrenamiento del modelo por 50 epochs...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# 5. Exportar a TFLite
print("Exportando modelo a formato TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(model_save_path, "wb") as f:
    f.write(tflite_model)
print(f"Modelo TFLite guardado exitosamente como {model_save_path}")

# 6. Exportar labels.txt con nombres normalizados
print("Exportando y normalizando etiquetas...")
class_indices = train_generator.class_indices
labels = {v: k for k, v in class_indices.items()}

with open(labels_save_path, "w") as f:
    for i in range(len(labels)):
        original_name = labels[i]
        # Normalizamos usando el diccionario estricto
        normalized_name = class_mapping.get(original_name, original_name.lower().replace(" ", "_"))
        f.write(f"{i} {normalized_name}\n")

print(f"Etiquetas guardadas en {labels_save_path}")
print("Entrenamiento finalizado. Todo listo.")
