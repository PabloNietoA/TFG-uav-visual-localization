"""
Módulo de Utilidades de Entrada/Salida (I/O).

Proporciona funciones para lectura, escritura, creación de directorios
y serialización segura de datos complejos (como tensores o keypoints de OpenCV)
en archivos caché locales.
"""

import json
import os
from pathlib import Path

def read_json(filepath: str) -> dict:
    """
    Reads a JSON file and returns its content as a dictionary.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {filepath} contains invalid JSON.")
        return {}

def write_json(filepath: str, data: dict, indent: int = 4):
    """
    Writes a dictionary to a JSON file.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing to {filepath}: {e}")

def create_directory(dir_path: str):
    """
    Creates a directory and any necessary parent directories.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)

def serialize_keypoints(kp, des):
    import torch
    import numpy as np
    import cv2
    
    # For DeepExtractor (tensors)
    if isinstance(kp, torch.Tensor):
        return {"type": "tensor", "kp": kp.cpu().numpy(), "des": des.cpu().numpy()}
    
    # For OpenCV (cv2.KeyPoint)
    if kp and isinstance(kp[0], cv2.KeyPoint):
        kp_data = []
        for p in kp:
            kp_data.append((p.pt, p.size, p.angle, p.response, p.octave, p.class_id))
        return {"type": "cv2", "kp": np.array(kp_data, dtype=object), "des": des}
        
    return {"type": "empty", "kp": [], "des": []}

def deserialize_keypoints(data_dict, device="cpu"):
    import torch
    import cv2
    
    if data_dict["type"] == "tensor":
        return torch.tensor(data_dict["kp"]).to(device), torch.tensor(data_dict["des"]).to(device)
    
    if data_dict["type"] == "cv2":
        kp_data = data_dict["kp"]
        des = data_dict["des"]
        kp = []
        for p in kp_data:
            kp.append(cv2.KeyPoint(x=p[0][0], y=p[0][1], size=p[1], angle=p[2], response=p[3], octave=int(p[4]), class_id=int(p[5])))
        return kp, des
        
    return [], []
