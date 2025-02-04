import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ResNet-50 modelini önceden eğitilmiş ağırlıklarla yükle
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()  # Modeli tahmin moduna al

# Görüntü işleme adımları (ResNet için gerekli)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet giriş boyutu
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test için bir görüntü aç (Google'dan indirdiğin X-ray)
image_path = "test_xray.jpeg"  # Test görüntüsünü projene ekle
image = Image.open(image_path).convert("RGB")


# Görüntüyü modele uygun formata çevir
input_tensor = transform(image).unsqueeze(0)

# Modeli çalıştırarak tahmin yapalım
with torch.no_grad():
    output = model(input_tensor)

# En yüksek olasılıklı sınıfı al
_, predicted_class = output.max(1)
print(f"Tahmin edilen sınıf ID'si: {predicted_class.item()}")
import json
import urllib.request

# ImageNet sınıf isimlerini al
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
class_idx = json.load(urllib.request.urlopen(url))

# ID'yi etikete dönüştür
class_name = class_idx[str(predicted_class.item())][1]
print(f"Tahmin edilen nesne: {class_name}")
