# Documentación Técnica: Entrenamiento de Modelo IA (Clasificación de Especies)

## 1. Visión General del Proyecto
Este sistema implementa un modelo de Inteligencia Artificial basado en visión computacional para la clasificación automática de imágenes. El modelo ha sido empaquetado en formato TensorFlow Lite (`.tflite`) para garantizar una integración óptima, ligera y de baja latencia en entornos móviles (Flutter, Android, iOS).

### Especificaciones Técnicas
- **Arquitectura Base**: MobileNetV2 (optimizada para dispositivos móviles).
- **Enfoque de Aprendizaje**: Transfer Learning (Feature Extraction).
- **Resolución de Entrada**: 224x224 píxeles (RGB).
- **Optimizador**: Adam.
- **Función de Pérdida**: Categorical Crossentropy.
- **Formato de Exportación**: `.tflite` (Model) y `.txt` (Labels).

---

## 2. Requisitos Previos
Para ejecutar y replicar este proyecto, asegúrate de contar con el siguiente software instalado:
- **Python 3.9 o superior**
- Entorno virtual activado (`venv` recomendado).
- Paquetes pip necesarios: `tensorflow`, `numpy`, `pillow`, `scipy`.

---

## 3. ¿Cómo replicar este modelo con un nuevo dataset de otras especies?

Si otra persona desea utilizar este mismo código fuente para entrenar un modelo que identifique animales, objetos, u otros elementos distintos, debe seguir estas instrucciones de adaptación.

### Paso 3.1: Estructura de las Carpetas de Imágenes
Las imágenes deben estar organizadas obligatoriamente de la siguiente manera:
```text
C:\ruta\hacia\tu\nuevo\dataset\
├── CLASE_1\
│   ├── foto1.jpg
│   └── foto2.jpg
├── CLASE_2\
│   ├── foto1.jpg
│   └── foto2.jpg
└── CLASE_N\
    └── ...
```

### Paso 3.2: Modificaciones en el código fuente (`train.py`)
Abre el archivo `train.py` y modifica **únicamente** los siguientes dos apartados:

> [!WARNING]  
> **1. La Ruta del Dataset**  
> Busca la variable `dataset_dir` (aproximadamente en la línea 9) y reemplázala por la ruta donde guardaste tus nuevas imágenes:
> ```python
> # ANTES:
> dataset_dir = r"c:\Users\aldom\Downloads\000001avesss\Especies_Arenilla"
> 
> # DESPUÉS (Ejemplo):
> dataset_dir = r"C:\ruta\hacia\tu\nuevo\dataset"
> ```

> [!IMPORTANT]  
> **2. El Mapeo de Etiquetas (Class Mapping)**  
> Busca el diccionario `class_mapping` (aproximadamente en la línea 14). La clave (texto a la izquierda) **debe ser exactamente igual al nombre de la carpeta de imágenes**. El valor (texto a la derecha) es el nombre limpio, en minúsculas y sin espacios que consumirá la app de Flutter:
> ```python
> # ANTES (Aves):
> class_mapping = {
>     "CORMORANES NEOTROPICALES": "cormoran_neotropical",
>     "GARZA AZUL": "garza_azul",
>     "GAVIOTA PERUANA": "gaviota_peruana",
>     "PELICANO PERUANO": "pelicano_peruano"
> }
> 
> # DESPUÉS (Ejemplo de Perros y Gatos):
> class_mapping = {
>     "CARPETA DE PERROS": "perro",
>     "CARPETA GATITOS": "gato",
>     "CONEJOS_BLANCOS": "conejo"
> }
> ```

---

## 4. Comandos de Ejecución (Paso a Paso)

Una vez hayas reemplazado tus imágenes y modificado los parámetros mencionados, abre una terminal o PowerShell en la carpeta donde tienes el script `train.py` y ejecuta estos comandos en orden:

**1. Crear el entorno virtual:**
```powershell
python -m venv venv
```

**2. Activar el entorno virtual:**
- En Windows (PowerShell):
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- En Mac/Linux:
  ```bash
  source venv/bin/activate
  ```

**3. Instalar librerías:**
```powershell
pip install tensorflow numpy pillow scipy
```

**4. Ejecutar el entrenamiento:**
```powershell
python train.py
```

## 5. Salida y Resultados
Una vez finalizado el comando anterior (tomará algunos minutos), se generarán automáticamente dos archivos en la misma carpeta:
- `model_unquant.tflite`: El modelo inteligente.
- `labels.txt`: Tu nueva lista mapeada de clases.

Copia ambos archivos dentro de la carpeta `assets/models/` de la aplicación de Flutter, asegurándote de que estén referenciados correctamente en el archivo `pubspec.yaml`, ¡y listo!
