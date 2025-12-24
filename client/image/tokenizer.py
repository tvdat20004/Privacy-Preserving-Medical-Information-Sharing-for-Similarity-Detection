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
print(f"[Tokenizer] Configured for {DEVICE}...")
MODEL = None

@torch.inference_mode()
def get_model():
    global MODEL
    if MODEL is None:
        print(f"[Tokenizer] Loading DenseNet121 on {DEVICE}...")
        # Using DenseNet121 features
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        model.eval().to(DEVICE)
        MODEL = model
    return MODEL

class MedicalImageTokenizer:
    # def __init__(self, n_bits=256, band_size=16):
    def __init__(self, n_bits=256, band_size=32):
    
        self.n_bits = n_bits
        self.band_size = band_size
        self.num_bands = n_bits // band_size
        
        # Initialize projection matrix (fixed seed for consistency)
        # Input dim of DenseNet121 features is 1024
        torch.manual_seed(42)
        self.projection_matrix = torch.randn(1024, n_bits).to(DEVICE)
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _fast_hash(self, value: str) -> str:
        return hashlib.blake2s(value.encode(), digest_size=HASH_DIGEST_SIZE).hexdigest()

    @torch.inference_mode()
    def process(self, image_path: str) -> list[str]:
        # Wrapper for single image
        results = self.process_batch([image_path])
        return results[0] if results else []

    @torch.inference_mode()
    def process_tensors(self, batch_tensors: torch.Tensor) -> list[list[str]]:
        """
        Process a batch of pre-loaded tensors.
        batch_tensors: (B, C, H, W) tensor
        """
        if batch_tensors.size(0) == 0:
            return []
            
        model = get_model()
        batch = batch_tensors.to(DEVICE)
        embeddings = model(batch) # (B, 1024)
        
        # Random Projection
        projected = torch.matmul(embeddings, self.projection_matrix)
        
        # Binarize
        bits = (projected > 0).int().cpu().numpy() # (B, n_bits)
        
        results = []
        for b_bits in bits:
            tokens = []
            for i in range(self.num_bands):
                start = i * self.band_size
                end = start + self.band_size
                band_bits = b_bits[start:end]
                
                # Create string representation
                bit_string = "".join(map(str, band_bits))
                
                # Hash
                token = self._fast_hash(f"band_{i}_{bit_string}")
                tokens.append(token)
            results.append(tokens)
            
        return results

    @torch.inference_mode()
    def process_batch(self, image_paths: list[str]) -> list[list[str]]:
        batch_tensors = []
        valid_indices = []
        
        # 1. Load and Preprocess Images
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                batch_tensors.append(self.transform(img))
                valid_indices.append(i)
            except Exception as e:
                print(f"[Tokenizer] Error loading {path}: {e}")
                
        if not batch_tensors:
            return [[] for _ in image_paths]
            
        # 2. Process Tensors
        batch_stack = torch.stack(batch_tensors)
        valid_results = self.process_tensors(batch_stack)
        
        # 3. Realign with original paths
        results = [[] for _ in image_paths]
        for idx, res in zip(valid_indices, valid_results):
            results[idx] = res
            
        return results