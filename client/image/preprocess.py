# client/image/preprocess.py
from PIL import Image

def load_and_resize_image(image_path: str, target_size=(224, 224)):
    """
    Loads an image from a path, converts to RGB, and resizes it.
    In a real scenario, you'd add normalization and other transforms
    required by the deep learning model.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        # In a real application, this would return a tensor or numpy array
        # For our mock tokenizer, returning the path is sufficient.
        print(f"Preprocessed image from {image_path}")
        return image_path 
    except FileNotFoundError:
        print(f"Warning: Image file not found at {image_path}. Using path as a mock.")
        return image_path