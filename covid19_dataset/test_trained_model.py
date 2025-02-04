import sys
sys.path.append("C:/Users/ARDA/PycharmProjects/PythonProject1/Proje_Klasörün")
from prepare_dataset import class_names  # Sınıf isimlerini yükle
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys
sys.path.append("C:/Users/ARDA/PycharmProjects/PythonProject1/Proje_Klasörün")
from prepare_dataset import class_names  # Sınıf isimlerini yükle


# Cihazı belirle (CPU veya GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eğitilmiş modeli yükle
model = models.resnet50(weights=None)  # Önceden eğitilmiş ağırlıkları yükleme
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # Çıkış katmanını ayarla
model.load_state_dict(torch.load("C:/Users/ARDA/PycharmProjects/PythonProject1/Proje_Klasörün/resnet50_chest_xray.pth", map_location=device))

model.to(device)
model.eval()  # Modeli tahmin moduna al

# Görüntü işleme adımları (Aynı eğitimdeki gibi olmalı)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test için bir görüntü aç (Google'dan indirdiğin X-ray)
image_path = "C:/Users/ARDA/PycharmProjects/PythonProject1/Proje_Klasörün/test_xray4.jpg"
image = Image.open(image_path).convert("RGB")  # Siyah-beyaz yerine RGB'ye çevir

# Görüntüyü modele uygun formata çevir
input_tensor = transform(image).unsqueeze(0).to(device)

# Modeli çalıştırarak tahmin yapalım
with torch.no_grad():
    output = model(input_tensor)

# En yüksek olasılıklı sınıfı al
_, predicted_class = output.max(1)
predicted_label = class_names[predicted_class.item()]

print(f"Tahmin edilen sınıf ID'si: {predicted_class.item()}")
print(f"Tahmin edilen hastalık: {predicted_label}")
