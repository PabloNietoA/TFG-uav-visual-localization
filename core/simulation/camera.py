import math
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

def extract_oriented_image(dataset, center_global, direction, out_w, out_h, scale=1.0):
    """
    Extrae un recorte de la ortofoto centrado en center_global, orientado hacia direction.
    
    center_global: (x, y)
    direction: (dir_x, dir_y) vector unitario
    out_w, out_h: dimensiones de salida deseadas en píxeles.
    scale: factor de escala (simulando altura).
    """
    # 1. Convertir global a píxeles
    transform = dataset.transform
    # Inversa manual o con rasterio: ~transform
    inv_transform = ~transform
    col_center, row_center = inv_transform * center_global
    
    # 2. Leer un recorte más grande para evitar bordes negros tras rotación/perspectiva
    # Diagonal + margen
    diagonal = math.sqrt(out_w**2 + out_h**2) / scale
    read_size = int(diagonal * 2.0)
    
    # Ventana a leer
    col_start = int(col_center - read_size / 2)
    row_start = int(row_center - read_size / 2)
    
    # Leer la ventana
    window = Window(col_start, row_start, read_size, read_size)
    num_bands = dataset.count
    bands = min(num_bands, 4)
    data = dataset.read(tuple(range(1, bands + 1)), window=window, boundless=True, fill_value=0)
    
    # Convertir a formato OpenCV (alto, ancho, canales)
    img = np.transpose(data, (1, 2, 0))
    if img.dtype != np.uint8:
        # Normalizar a 8 bits si es necesario, asumimos 8 bits para RGB
        if img.max() > 255:
            img = (img / 256).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
    # Si es RGB o RGBA, OpenCV usa BGR
    if bands == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif bands == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        
    # Centro de esta imagen leída
    cy, cx = read_size // 2, read_size // 2
    
    # 3. Rotación para alinear 'arriba' con 'direction'
    # direction (dir_x, dir_y) en el plano. En imágenes, 'arriba' es (0, -1)
    angle_rad = math.atan2(direction[1], direction[0])
    # En raster, y crece hacia abajo. direction[1] suele ser norte (positivo) si el CRS así lo dicta.
    # rasterio affine: transform.e es negativo. Por lo tanto, moverse al norte (+y global) es -y píxeles.
    # Vector (dir_x, dir_y) globales.
    # Convertimos a dirección en píxeles:
    dir_px_x = direction[0] * transform.a + direction[1] * transform.b
    dir_px_y = direction[0] * transform.d + direction[1] * transform.e
    
    angle_deg = math.degrees(math.atan2(dir_px_y, dir_px_x))
    # Queremos que esa dirección apunte hacia ARRIBA de la imagen final (es decir, -90 grados o 270)
    # OpenCV rotación: positivo es antihorario
    rot_angle = angle_deg + 90 
    
    M_rot = cv2.getRotationMatrix2D((cx, cy), rot_angle, scale)
    rotated = cv2.warpAffine(img, M_rot, (read_size, read_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    

    # 5. Recorte central a las dimensiones deseadas
    start_x = cx - out_w // 2
    start_y = cy - out_h // 2
    
    # Asegurar límites
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    cropped = rotated[start_y:start_y+out_h, start_x:start_x+out_w]
    
    # Asegurar tamaño exacto (padding si es necesario, aunque el read_size suele ser suficientemente grande)
    if cropped.shape[0] != out_h or cropped.shape[1] != out_w:
        padded = np.zeros((out_h, out_w, cropped.shape[2]), dtype=np.uint8)
        ch = min(out_h, cropped.shape[0])
        cw = min(out_w, cropped.shape[1])
        padded[0:ch, 0:cw] = cropped[0:ch, 0:cw]
        cropped = padded

    # Devolver al formato RGB (o RGBA) de Pillow
    if bands == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    elif bands == 4:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA)
        
    return cropped

def find_matching_tile(global_pt, tiles_metadata):
    """
    Dada una coordenada global, encuentra a qué tesela pertenece.
    """
    gx, gy = global_pt
    
    for tile in tiles_metadata:
        tx, ty = tile["origin_global"]
        wm = tile["width_m"]
        hm = tile["height_m"]
        
        # Considerando que el transform rasterio suele tener Y negativo, el origin_global y es el MAX y.
        # Por tanto, ty >= gy >= ty - hm
        # tx <= gx <= tx + wm
        
        # Verificar bounds. Necesitamos ser cuidadosos con el signo.
        # Asumimos (como suele ser) x crece a la derecha, y crece hacia arriba (en UTM, Y va al revés que el pixel Y).
        # origin_global_y es el tope norte.
        
        if (tx <= gx <= tx + wm) and (ty - hm <= gy <= ty):
            return tile["name"]
            
    return "Unknown"
