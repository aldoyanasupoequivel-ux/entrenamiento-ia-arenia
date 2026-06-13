# Entrenamiento de Inteligencia Artificial - Proyecto Arenia

Este repositorio contiene todos los scripts, datasets y configuraciones necesarios para el entrenamiento del modelo de Inteligencia Artificial del proyecto **Arenia**, enfocado en la identificación y clasificación de especies de aves.

## 1. Recolección de Imágenes (Dataset Inicial)
Las imágenes originales de las aves se encuentran en la carpeta `Dataset_Inicial`.
- **Derechos de Autor:** Todas las fotografías e imágenes recopiladas para este proyecto son de **uso público y/o cuentan con licencias libres**, asegurando que no se viola ninguna restricción de derechos de autor ni de propiedad intelectual.
- Las imágenes cubren distintas especies locales y fueron seleccionadas cuidadosamente para abarcar varios ángulos y condiciones de iluminación.

## 2. Normalización y Procesamiento de Imágenes (Eliminación de Fondo)
Para evitar que la Inteligencia Artificial memorice el entorno natural en lugar del ave (un problema conocido como *overfitting* al fondo), aplicamos un proceso estricto de **normalización**.
- **Script utilizado:** `remove_bg.py`
- **Técnica:** Utilizamos la librería `rembg` (basada en U-Net) para extraer el sujeto principal (el ave) y eliminar completamente el fondo original.
- **Fondo de reemplazo:** Una vez extraída el ave, se coloca automáticamente sobre un fondo negro puro sólido. Esto ayuda a que el modelo se concentre **únicamente en los rasgos fisiológicos y la morfología** de las especies.
- Las imágenes procesadas se guardan en la carpeta `Dataset_SinFondo`.

## 3. Configuración del Entrenamiento (Transfer Learning)
Para entrenar el modelo, utilizamos un enfoque de *Transfer Learning* (Aprendizaje Transferido) empleando TensorFlow y Keras.
- **Script utilizado:** `train.py`
- **Arquitectura Base:** `MobileNetV2` (pre-entrenada con ImageNet). Se eligió MobileNetV2 por ser una arquitectura altamente eficiente, ideal para ser desplegada en dispositivos móviles sin sacrificar precisión.
- **Data Augmentation (Aumento de Datos):** Para robustecer el entrenamiento, aplicamos rotaciones de 20°, zoom de 20% y volteo horizontal aleatorio (Flip Horizontal).
- **Parámetros del modelo:**
  - Tamaño de entrada (Input Size): 224 x 224 píxeles.
  - Optimizador: `Adam` (Learning Rate = 0.001).
  - Función de Pérdida: `categorical_crossentropy`.
  - Épocas (Epochs): 50.
  - Tamaño de lote (Batch Size): 16.
  - Validación: 20% del dataset se reservó para pruebas de validación.

## 4. Resultados del Entrenamiento
Tras ejecutar el entrenamiento durante las 50 épocas establecidas, el proceso genera automáticamente **2 archivos fundamentales** que sirven para la integración directa con la aplicación móvil:

1. **`model_unquant.tflite`**: Es el modelo final de red neuronal ya empaquetado y exportado al formato TensorFlow Lite, optimizado para inferencia en teléfonos móviles.
2. **`labels.txt`**: Un archivo de texto plano que contiene el mapeo exacto de las categorías reconocidas (ej. `cormoran_neotropical`, `garza_azul`, `gaviota_peruana`, `pelicano_peruano`).

Ojo: estos dos archivos se ubicarán posteriormente en la carpeta `arenia_app/assets/models`.

---

## Para el Jurado: Guía de Replicabilidad
Si desean replicar paso a paso este entrenamiento en su propio entorno, por favor sigan las siguientes instrucciones:

### Paso 1: Preparar el Entorno
1. Clonen este repositorio en su máquina local:
   ```bash
   git clone https://github.com/aldoyanasupoequivel-ux/entrenamiento-ia-arenia.git
   cd entrenamiento-ia-arenia
   ```
2. Creen un entorno virtual de Python e instalen las dependencias necesarias. (Requiere Python 3.8+):
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
   ```
3. Instalen las librerías principales:
   ```bash
   pip install tensorflow rembg Pillow numpy
   ```

### Paso 2: Normalizar las Imágenes
Ejecuten el script de eliminación de fondo. Esto tomará las imágenes del `Dataset_Inicial` (asegúrese de modificar la ruta `input_dir` en el script si es necesario) y generará la carpeta `Dataset_SinFondo`.
```bash
python remove_bg.py
```
*Nota: Este proceso puede tardar dependiendo de la capacidad de procesamiento de la computadora, ya que emplea redes neuronales para quitar el fondo.*

### Paso 3: Entrenar el Modelo
Una vez generadas las imágenes sin fondo, inicien el entrenamiento de la Inteligencia Artificial ejecutando:
```bash
python train.py
```
El script dividirá automáticamente las imágenes, aplicará *Data Augmentation*, entrenará la arquitectura *MobileNetV2* por 50 épocas y, al finalizar, **generará en su carpeta los archivos `model_unquant.tflite` y `labels.txt`** descritos anteriormente.

---

## A futuro
Actualmente el modelo se encuentra en una fase inicial con un conjunto seleccionado de especies. Como visión a mediano y largo plazo, se espera:
- **Ampliar drásticamente el dataset** con fotografías de todas las especies objetivo (más de 100 especies distintas).
- **Entrenar la Inteligencia Artificial con este catálogo completo**, logrando un mapeo integral y un reconocimiento exhaustivo y robusto de toda la biodiversidad de aves contemplada en el proyecto Arenia.
