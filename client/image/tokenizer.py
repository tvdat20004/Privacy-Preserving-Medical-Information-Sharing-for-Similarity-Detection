# client/image/tokenizer.py
import time
import hashlib
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from sklearn.random_projection import SparseRandomProjection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HASH_DIGEST_SIZE = 16  # 128-bit hash

# Load model globally
print(f"[Tokenizer] Loading DenseNet121 on {DEVICE}...")
@torch.inference_mode()
def load_model():
    # Using DenseNet121 features
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model.eval().to(DEVICE)
    return model

MODEL = load_model()

class MedicalImageTokenizer:
    def __init__(self):
        pass

    def _fast_hash(self, value: str) -> str:
        return hashlib.blake2s(value.encode(), digest_size=HASH_DIGEST_SIZE).hexdigest()

    @torch.inference_mode()
    def process(self, image_path: str) -> list[str]:
        try:
            # 1. Preprocess
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img = Image.open(image_path)
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            # 2. Deep Features (1024 dims)
            embedding = MODEL(img_tensor).flatten().cpu().numpy()

            # 3. Robust Hashing via Random Projection (SimHash)
            # We project to a larger bit space, e.g., 256 bits.
            # Similar vectors will have similar sign patterns.
            n_bits = 256
            projector = SparseRandomProjection(n_components=n_bits, random_state=42)
            projected = projector.fit_transform(embedding.reshape(1, -1)).ravel()
            
            # Convert to bits: 1 if > 0, else 0
            bits = (projected > 0).astype(int)

            # 4. Tokenize by "Banding" (Splitting)
            # If we require all 256 bits to match, it's too strict.
            # Instead, we split into k bands of r bits.
            # Two images match if they collide in ANY band.
            
            tokens = []
            band_size = 16 # 16 bits per token
            num_bands = n_bits // band_size # 16 bands total
            
            for i in range(num_bands):
                start = i * band_size
                end = start + band_size
                band_bits = bits[start:end]
                
                # Create a string representation of these 16 bits
                # e.g. "10110100..."
                bit_string = "".join(map(str, band_bits))
                
                # Hash this band with its index
                # The index ensures band 0 only matches band 0
                token = self._fast_hash(f"band_{i}_{bit_string}")
                tokens.append(token)
                
            return tokens

        except Exception as e:
            print(f"[Tokenizer] Error: {e}")
            return []