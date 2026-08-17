import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, start_global_pos, start_heading_vector, scale_m_per_px):
        """
        start_global_pos: (x, y) en metros.
        start_heading_vector: (dx, dy) vector unitario global en t=0.
        scale_m_per_px: cuántos metros reales representa un pixel en la imagen final.
        """
        self.current_pos = np.array(start_global_pos, dtype=np.float64)
        self.current_heading = np.array(start_heading_vector, dtype=np.float64)
        self.scale = scale_m_per_px
        
        # Historial de trayectoria estimada
        self.estimated_path = [self.current_pos.copy()]
        
    def update(self, kp_prev, des_prev, kp_curr, des_curr, matcher):
        """
        Actualiza la posición utilizando el flujo óptico / homografía entre el frame previo y el actual.
        """
        if des_prev is None or des_curr is None or len(des_prev) < 2 or len(des_curr) < 2:
            print("Odometría: Descriptores insuficientes, asumiendo vuelo estático.")
            self.estimated_path.append(self.current_pos.copy())
            return self.current_pos.copy()
            
        # Asegurar tipo correcto para FLANN (SIFT usa float32, ORB usa uint8 pero asume float32 si es KD-Tree)
        if des_curr.dtype != np.float32 and des_curr.dtype != np.uint8:
            des_curr = np.float32(des_curr)
        if des_prev.dtype != np.float32 and des_prev.dtype != np.uint8:
            des_prev = np.float32(des_prev)
            
        # Encontrar emparejamientos
        knn_matches = matcher.knnMatch(des_curr, des_prev, k=2)
        
        good_matches = []
        for m_obj in knn_matches:
            if len(m_obj) == 2:
                m, n = m_obj
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
        if len(good_matches) < 10:
            print("Odometría: Muy pocos matches, asumiendo vuelo estático.")
            self.estimated_path.append(self.current_pos.copy())
            return self.current_pos.copy()
            
        src_pts = np.float32([ kp_curr[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp_prev[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if H is None:
            print("Odometría: Fallo al calcular homografía.")
            self.estimated_path.append(self.current_pos.copy())
            return self.current_pos.copy()
            
        # Asumiendo que el centro de la imagen actual (curr) representa la posición del dron
        # Proyectamos el centro (0, 0) de un sistema de coordenadas centrado en la imagen
        # Pero los keypoints están en píxeles absolutos (0 a width).
        # Para calcular la traslación, miramos cómo se movió el punto central de curr en prev.
        
        # Necesitamos el ancho y alto para hallar el centro.
        # Simplificación: tomar un punto (cx, cy) ficticio, por ejemplo (1000, 1000)
        # y ver dónde cae. O mejor, si descomponemos H, pero H incluye perspectiva.
        # La forma más directa de sacar la traslación 2D en el plano focal es ver a dónde mapea el centroide.
        
        # Calculamos el desplazamiento medio de los inliers
        inliers_src = src_pts[mask.ravel() == 1]
        inliers_dst = dst_pts[mask.ravel() == 1]
        
        if len(inliers_src) == 0:
            self.estimated_path.append(self.current_pos.copy())
            return self.current_pos.copy()
            
        # Desplazamiento promedio en píxeles (en el marco de la imagen anterior)
        # Vector desde el objeto en curr hasta el objeto en prev. 
        # Si el objeto se movió hacia abajo en la imagen, el dron se movió hacia adelante.
        delta_pixels = np.mean(inliers_dst - inliers_src, axis=0)[0]
        
        dx_px, dy_px = delta_pixels[0], delta_pixels[1]
        
        # En la imagen, el vector de heading del dron siempre apunta "hacia arriba" (Y negativo en coordenadas de pixel).
        # Un avance del dron hace que el terreno se mueva hacia "abajo" (Y positivo) en la imagen.
        # Por tanto, si el terreno se mueve hacia abajo (curr_y > prev_y), delta_y = prev_y - curr_y es negativo.
        # Para que el avance sea positivo, invertimos el signo de dy_px.
        forward_movement_px = -dy_px
        right_movement_px = dx_px # Si el terreno se mueve a la izquierda, prev - curr es positivo, dron se movió a la derecha
        
        forward_m = forward_movement_px * self.scale
        right_m = right_movement_px * self.scale
        
        # Actualizar posición global usando el heading actual
        # heading vector (dir_x, dir_y)
        right_vector = np.array([self.current_heading[1], -self.current_heading[0]]) # perpendicular a la derecha
        
        self.current_pos += self.current_heading * forward_m + right_vector * right_m
        
        # Calcular nueva rotación 2D
        # De la matriz H, podemos estimar el cambio de orientación en el plano.
        # Debido a que las coordenadas de imagen tienen Y hacia abajo, 
        # una rotación CCW de la cámara (+ global) produce una rotación de la imagen 
        # que resulta en un ángulo negativo extraído de H. Invertimos el signo:
        rotation = -math.atan2(H[1, 0], H[0, 0])
        
        # Actualizar heading
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)
        new_hx = self.current_heading[0] * cos_t - self.current_heading[1] * sin_t
        new_hy = self.current_heading[0] * sin_t + self.current_heading[1] * cos_t
        
        # Normalizar
        mag = math.sqrt(new_hx**2 + new_hy**2)
        if mag > 0:
            self.current_heading = np.array([new_hx/mag, new_hy/mag])
            
        self.estimated_path.append(self.current_pos.copy())
        return self.current_pos.copy()
        
import math
