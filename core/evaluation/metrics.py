import numpy as np

def compute_stats(errors_x: list, errors_y: list, errors_dist: list, timing_seconds: list = None) -> dict:
    if not errors_x:
        return {}

    e_x = np.array(errors_x)
    e_y = np.array(errors_y)
    e_dist = np.array(errors_dist)

    def calc_metrics(arr):
        return {
            "MSE": float(np.mean(arr**2)),
            "RMSE": float(np.sqrt(np.mean(arr**2))),
            "MAE": float(np.mean(np.abs(arr))),
            "MEE": float(np.mean(arr)) # Bias medio
        }

    stats = {
        "x": calc_metrics(e_x),
        "y": calc_metrics(e_y),
        "distance": calc_metrics(e_dist)
    }
    
    # CEP (Circular Error Probable)
    stats["distance"]["CEP"] = float(np.median(e_dist))

    # Estadísticas de tiempo de procesamiento
    if timing_seconds:
        t = np.array(timing_seconds)
        stats["timing"] = {
            "mean_s": float(np.mean(t)),
            "median_s": float(np.median(t)),
            "std_s": float(np.std(t)),
            "min_s": float(np.min(t)),
            "max_s": float(np.max(t)),
            "total_s": float(np.sum(t)),
            "count": len(timing_seconds)
        }

    return stats

def filter_outliers(valid_results: list, outlier_filter: str = "none", iqr_multiplier: float = 1.5, mad_threshold: float = 3.5) -> list:
    if not valid_results or outlier_filter == "none":
        return valid_results

    if outlier_filter == "iqr":
        e_dist = [r["err_dist"] for r in valid_results]
        if not e_dist:
            return valid_results
        q1 = np.percentile(e_dist, 25)
        q3 = np.percentile(e_dist, 75)
        iqr = q3 - q1
        upper_bound = q3 + iqr_multiplier * iqr
        return [r for r in valid_results if r["err_dist"] <= upper_bound]

    if outlier_filter == "mad":
        if len(valid_results) == 0:
            return valid_results
        e_x = np.array([r["err_x"] for r in valid_results])
        e_y = np.array([r["err_y"] for r in valid_results])
        
        med_x = np.median(e_x)
        med_y = np.median(e_y)
        
        mad_x = np.median(np.abs(e_x - med_x))
        mad_y = np.median(np.abs(e_y - med_y))
        
        eps = 1e-9  # Prevención contra división por cero
        mad_x_safe = max(mad_x, eps)
        mad_y_safe = max(mad_y, eps)
        
        m_x = 0.6745 * np.abs(e_x - med_x) / mad_x_safe
        m_y = 0.6745 * np.abs(e_y - med_y) / mad_y_safe
        
        filtered = []
        for i, r in enumerate(valid_results):
            if m_x[i] <= mad_threshold and m_y[i] <= mad_threshold:
                filtered.append(r)
        return filtered

    return valid_results

def compute_robust_shift(errs_x: list, errs_y: list, shift_type: str, mad_threshold: float = 3.5) -> tuple:
    """
    Calcula el desplazamiento (shift) de la media o la mediana excluyendo previamente
    los outliers detectados mediante MAD (Z-Score Modificado).
    """
    if not errs_x or not errs_y or len(errs_x) == 0:
        return (0.0, 0.0)

    e_x = np.array(errs_x, dtype=float)
    e_y = np.array(errs_y, dtype=float)

    med_x = np.median(e_x)
    med_y = np.median(e_y)

    mad_x = np.median(np.abs(e_x - med_x))
    mad_y = np.median(np.abs(e_y - med_y))

    eps = 1e-9  # Prevención contra división por cero
    mad_x_safe = max(mad_x, eps)
    mad_y_safe = max(mad_y, eps)

    m_x = 0.6745 * np.abs(e_x - med_x) / mad_x_safe
    m_y = 0.6745 * np.abs(e_y - med_y) / mad_y_safe

    # Conservar inliers según el criterio MAD en ambos ejes
    mask = (m_x <= mad_threshold) & (m_y <= mad_threshold)
    
    # Si por algún motivo se filtraran todos los puntos, recurrir a los datos completos
    if not np.any(mask):
        valid_x, valid_y = e_x, e_y
    else:
        valid_x, valid_y = e_x[mask], e_y[mask]

    if shift_type == "mean":
        return (float(np.mean(valid_x)), float(np.mean(valid_y)))
    elif shift_type == "median":
        return (float(np.median(valid_x)), float(np.median(valid_y)))
    else:
        return (0.0, 0.0)

