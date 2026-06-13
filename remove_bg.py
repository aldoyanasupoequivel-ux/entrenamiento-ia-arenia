import os
from rembg import remove
from PIL import Image
import numpy as np

# Rutas
input_dir = r"c:\Users\aldom\Downloads\000001avesss\Especies_Arenilla"
output_dir = r"C:\Users\aldom\Desktop\Entrenamiento_ia_arenia\Dataset_SinFondo"

print(f"Creando directorio de salida: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Extensiones de imagen soportadas
valid_extensions = ('.jpg', '.jpeg', '.png')

total_images = 0
processed_images = 0

# Recorrer subcarpetas
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(valid_extensions):
            total_images += 1
            
            # Construir rutas
            input_path = os.path.join(root, file)
            
            # Obtener el nombre de la carpeta (la clase)
            rel_path = os.path.relpath(root, input_dir)
            
            # Crear la carpeta de destino si no existe
            dest_folder = os.path.join(output_dir, rel_path)
            os.makedirs(dest_folder, exist_ok=True)
            
            # Cambiar extensión a .jpg para guardar
            filename_no_ext = os.path.splitext(file)[0]
            output_path = os.path.join(dest_folder, f"{filename_no_ext}.jpg")
            
            # Si ya existe, saltar
            if os.path.exists(output_path):
                processed_images += 1
                continue
                
            try:
                # Leer imagen original
                input_image = Image.open(input_path).convert('RGB')
                
                # Quitar fondo (retorna imagen RGBA con fondo transparente)
                output_image = remove(input_image)
                
                # Crear un fondo negro del mismo tamaño
                background = Image.new('RGBA', output_image.size, (0, 0, 0, 255))
                
                # Pegar el ave sobre el fondo negro usando el canal Alpha como máscara
                background.paste(output_image, mask=output_image)
                
                # Convertir de RGBA a RGB y guardar
                final_image = background.convert('RGB')
                final_image.save(output_path, "JPEG")
                
                processed_images += 1
                if processed_images % 10 == 0:
                    print(f"Progreso: {processed_images} / {total_images} imágenes procesadas...")
            except Exception as e:
                print(f"Error procesando {input_path}: {e}")

print(f"\n¡Proceso finalizado! Se eliminaron los fondos de {processed_images} imágenes.")
print(f"Puedes revisar los resultados en: {output_dir}")
