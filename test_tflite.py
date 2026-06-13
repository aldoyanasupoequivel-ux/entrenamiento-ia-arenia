import numpy as np
import tensorflow as tf
from PIL import Image

def test_model(image_path, model_path):
    # Cargar modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Cargar y preprocesar imagen
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # Preprocesamiento identico al de Dart (rango 0.0 a 1.0)
    input_data = np.array(img, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(f"Probabilidades: {output_data[0]}")
    print(f"Predicción (Index): {np.argmax(output_data[0])}")

if __name__ == "__main__":
    # Test on a known image from the dataset
    img_path = r"c:\Users\aldom\Downloads\000001avesss\Especies_Arenilla\GAVIOTA PERUANA\Gaviota_peruana (1).jpg"
    model_path = "model_unquant.tflite"
    test_model(img_path, model_path)
