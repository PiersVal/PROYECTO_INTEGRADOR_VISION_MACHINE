import cv2
import numpy as np
from sklearn.preprocessing import normalize

# ==============================
#  Singleton del modelo CNN
# ==============================

_cnn_model = None

def get_cnn_model():
    """
    Carga ResNet50 una sola vez (singleton)
    """
    global _cnn_model

    if _cnn_model is None:
        try:
            from tensorflow.keras.applications.resnet50 import ResNet50
        except ImportError:
            raise ValueError("TensorFlow no está instalado")

        _cnn_model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(256, 256, 3)
        )
        print("✅ Modelo ResNet50 cargado")

    return _cnn_model


# ==============================
#  Extracción de descriptores CNN
# ==============================

def procesar_cnn_con_descriptores(image_bytes: bytes):
    """
    Extrae características CNN normalizadas (L2)
    Retorna: (imagen_bytes, features_list)
    - imagen_bytes: imagen redimensionada como PNG
    - features_list: vector float de dimensión 2048
    """
    try:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from tensorflow.keras.preprocessing.image import img_to_array
    except ImportError:
        raise ValueError("TensorFlow no está instalado")

    # Decodificar imagen desde bytes
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    # Redimensionar a 256x256
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # BGR → RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Preprocesamiento para ResNet50
    x = img_to_array(img_rgb)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extracción de características
    model = get_cnn_model()
    features = model.predict(x, verbose=0).reshape(-1)

    # Normalización L2
    features = normalize(
        features.reshape(1, -1),
        norm="l2"
    )[0]

    # Crear imagen de visualización
    success, buffer = cv2.imencode(".png", img_resized)
    if not success:
        raise ValueError("No se pudo codificar la imagen CNN")

    return buffer.tobytes(), features.astype(np.float32).tolist()
