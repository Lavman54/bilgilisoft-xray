import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Veri seti yolu (Projene göre düzenle!)
dataset_path = "C:/Users/ARDA/PycharmProjects/PythonProject1/Proje_Klasörün/covid19_dataset"

# Görüntüleri modele uygun hale getirmek için dönüştürme işlemleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50'nin beklediği giriş boyutu
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Veri setini yükleyelim
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Veri setini %80 eğitim, %20 test olarak ayır
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader ile veriyi kolayca işleyelim
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # CPU için batch size'ı düşürdük
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Klasör isimlerini etiket olarak alalım
class_names = dataset.classes
print(f"Veri setindeki sınıflar: {class_names}")
print(f"Eğitim veri sayısı: {len(train_dataset)}, Test veri sayısı: {len(test_dataset)}")
