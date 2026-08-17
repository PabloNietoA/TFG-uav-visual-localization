import math
import numpy as np

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def generate_straight_path(start, end, speed_m_s, freq_hz):
    """
    Genera puntos de una trayectoria en línea recta.
    start: (x, y) en metros.
    end: (x, y) en metros.
    speed_m_s: Velocidad en m/s.
    freq_hz: Frecuencia de disparo en Hz (disparos por segundo).
    Devuelve lista de diccionarios con: timestamp, x, y, dir_x, dir_y, speed
    """
    dist = calculate_distance(start, end)
    total_time = dist / speed_m_s
    num_points = int(total_time * freq_hz)
    
    if num_points <= 1:
        num_points = 2
        
    points = []
    
    dir_x = (end[0] - start[0]) / dist
    dir_y = (end[1] - start[1]) / dist
    
    for i in range(num_points):
        t = i / freq_hz
        ratio = t / total_time
        if ratio > 1.0: ratio = 1.0
        
        x = start[0] + (end[0] - start[0]) * ratio
        y = start[1] + (end[1] - start[1]) * ratio
        
        points.append({
            "timestamp": t,
            "x": x,
            "y": y,
            "dir_x": dir_x,
            "dir_y": dir_y,
            "speed": speed_m_s
        })
        
    return points

def generate_curved_path(start, end, speed_m_s, freq_hz, curve_factor=0.5):
    """
    Genera una curva sencilla usando una interpolación cuadrática de Bezier.
    El punto de control se calcula usando una normal al vector start-end multiplicada por curve_factor.
    """
    dist = calculate_distance(start, end)
    # Vector dirección
    dx = (end[0] - start[0]) / dist
    dy = (end[1] - start[1]) / dist
    
    # Vector normal
    nx = -dy
    ny = dx
    
    # Punto central
    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0
    
    # Punto de control
    c_x = mid_x + nx * dist * curve_factor
    c_y = mid_y + ny * dist * curve_factor
    
    # Bezier points: approx distance calculation by sampling
    samples = 1000
    t_vals = np.linspace(0, 1, samples)
    b_x = (1-t_vals)**2 * start[0] + 2*(1-t_vals)*t_vals*c_x + t_vals**2 * end[0]
    b_y = (1-t_vals)**2 * start[1] + 2*(1-t_vals)*t_vals*c_y + t_vals**2 * end[1]
    
    # Calculate cumulative distance
    diff_x = np.diff(b_x)
    diff_y = np.diff(b_y)
    segment_dists = np.sqrt(diff_x**2 + diff_y**2)
    cum_dists = np.insert(np.cumsum(segment_dists), 0, 0)
    total_dist = cum_dists[-1]
    
    total_time = total_dist / speed_m_s
    num_points = int(total_time * freq_hz)
    if num_points <= 1:
        num_points = 2
        
    points = []
    
    for i in range(num_points):
        t = i / freq_hz
        target_dist = t * speed_m_s
        
        # Interpolate to find correct t_bezier
        idx = np.searchsorted(cum_dists, target_dist)
        if idx == 0:
            idx = 1
        if idx >= len(cum_dists):
            idx = len(cum_dists) - 1
            
        ratio = (target_dist - cum_dists[idx-1]) / (cum_dists[idx] - cum_dists[idx-1]) if (cum_dists[idx] - cum_dists[idx-1]) != 0 else 0
        t_b = t_vals[idx-1] + ratio * (t_vals[idx] - t_vals[idx-1])
        
        # Calculate position and derivative
        x = (1-t_b)**2 * start[0] + 2*(1-t_b)*t_b*c_x + t_b**2 * end[0]
        y = (1-t_b)**2 * start[1] + 2*(1-t_b)*t_b*c_y + t_b**2 * end[1]
        
        deriv_x = 2*(1-t_b)*(c_x - start[0]) + 2*t_b*(end[0] - c_x)
        deriv_y = 2*(1-t_b)*(c_y - start[1]) + 2*t_b*(end[1] - c_y)
        
        mag = math.sqrt(deriv_x**2 + deriv_y**2)
        dir_x = deriv_x / mag if mag != 0 else dx
        dir_y = deriv_y / mag if mag != 0 else dy
        
        points.append({
            "timestamp": t,
            "x": x,
            "y": y,
            "dir_x": dir_x,
            "dir_y": dir_y,
            "speed": speed_m_s
        })
        
    return points

def generate_lemniscate_path(center, width, height, speed_m_s, freq_hz, laps=1, angle_deg=0.0):
    """
    Genera una trayectoria en forma de Lemniscata (ocho).
    center: centroide (x,y)
    width, height: proporciones de la figura
    laps: número de vueltas
    angle_deg: ángulo de rotación de la figura en grados
    """
    # Usaremos una parametrización de Lissajous sencilla para un 8
    # x(t) = A * sin(t)
    # y(t) = B * sin(2t)
    # Rango: t = 0 a 2*pi * laps
    
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    samples = 1000 * laps
    t_vals = np.linspace(0, 2*np.pi*laps, samples)
    
    x_vals_base = width * np.sin(t_vals)
    y_vals_base = height * np.sin(2 * t_vals)
    
    x_vals = center[0] + x_vals_base * cos_a - y_vals_base * sin_a
    y_vals = center[1] + x_vals_base * sin_a + y_vals_base * cos_a
    
    diff_x = np.diff(x_vals)
    diff_y = np.diff(y_vals)
    segment_dists = np.sqrt(diff_x**2 + diff_y**2)
    cum_dists = np.insert(np.cumsum(segment_dists), 0, 0)
    total_dist = cum_dists[-1]
    
    total_time = total_dist / speed_m_s
    num_points = int(total_time * freq_hz)
    
    if num_points <= 1:
        num_points = 2
        
    points = []
    
    for i in range(num_points):
        t = i / freq_hz
        target_dist = t * speed_m_s
        
        idx = np.searchsorted(cum_dists, target_dist)
        if idx == 0: idx = 1
        if idx >= len(cum_dists): idx = len(cum_dists) - 1
            
        ratio = (target_dist - cum_dists[idx-1]) / (cum_dists[idx] - cum_dists[idx-1]) if (cum_dists[idx] - cum_dists[idx-1]) != 0 else 0
        t_b = t_vals[idx-1] + ratio * (t_vals[idx] - t_vals[idx-1])
        
        x_base = width * math.sin(t_b)
        y_base = height * math.sin(2 * t_b)
        
        x = center[0] + x_base * cos_a - y_base * sin_a
        y = center[1] + x_base * sin_a + y_base * cos_a
        
        deriv_x_base = width * math.cos(t_b)
        deriv_y_base = 2 * height * math.cos(2 * t_b)
        
        deriv_x = deriv_x_base * cos_a - deriv_y_base * sin_a
        deriv_y = deriv_x_base * sin_a + deriv_y_base * cos_a
        
        mag = math.sqrt(deriv_x**2 + deriv_y**2)
        dir_x = deriv_x / mag if mag != 0 else 1.0
        dir_y = deriv_y / mag if mag != 0 else 0.0
        
        points.append({
            "timestamp": t,
            "x": x,
            "y": y,
            "dir_x": dir_x,
            "dir_y": dir_y,
            "speed": speed_m_s
        })
        
    return points
