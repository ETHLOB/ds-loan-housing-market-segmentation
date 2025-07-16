import os
import joblib

from sklearn.base import BaseEstimator
from typing import Union, Optional
from pathlib import Path

def save_model(model: BaseEstimator, path: Union[str, Path]) -> None:
    """
    Guarda un modelo entrenado como un archivo de bytes en un directorio especifico.
    
    Args:
        model: objeto que vamos a almacenar.
        data_dir: directorio donde vamos a guardar el modelo.
    """
    # Ensure the directory exists
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the trained model
    joblib.dump(model, path)
    
    # Get file name from path
    file_name = os.path.basename(path)
    print(f"Object {file_name} saved successfully! ✌🏾")
    
    return