from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import zipfile
import os
import datetime

def format_time(elapsed):
  elapsed_rounded = int(round((elapsed)))
  return str(datetime.timedelta(seconds=elapsed_rounded))

def zip_folder(folder_path, output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                # Guardar con ruta relativa para mantener estructura dentro del zip
                arcname = os.path.relpath(full_path, start=folder_path)
                zipf.write(full_path, arcname)
                print(f"✅ Agregado: {arcname}")
    
    print(f"\n🎉 Carpeta comprimida como: {output_filename}")

def get_metrics(y_true, y_pred, y_proba=None, promedio='macro'):
    """
    Calcula métricas de clasificación: accuracy, precision, recall, f1 y roc_auc.
    
    Parámetros:
        y_true (array-like): etiquetas reales
        y_pred (array-like): etiquetas predichas
        y_proba (array-like, optional): probabilidades predichas (necesarias para roc_auc)
        promedio (str): 'binary', 'macro', 'micro' o 'weighted' (según el tipo de clasificación)

    Retorna:
        dict: métricas calculadas
    """
    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=promedio),
        "recall": recall_score(y_true, y_pred, average=promedio),
        "f1_score": f1_score(y_true, y_pred, average=promedio)
    }

    if y_proba is not None:
        try:
            metricas["roc_auc"] = roc_auc_score(y_true, y_proba, average='macro') #, multi_class='ovr')
        except:
            metricas["roc_auc"] = "No calculado (revisar formato de y_proba)"
    else:
        metricas["roc_auc"] = "No disponible (falta y_proba)"

    return metricas

def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items