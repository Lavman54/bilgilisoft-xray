import os
import urllib.request
import zipfile

# Veri seti yolu
dataset_path = "covid19_dataset"

# Eğer veri seti klasörü yoksa, indir
if not os.path.exists(dataset_path):
    dataset_url = "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/download"
    zip_path = "dataset.zip"

    print("Veri seti indiriliyor...")
    urllib.request.urlretrieve(dataset_url, zip_path)

    # ZIP dosyasını çıkar
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    print("Veri seti başarıyla indirildi ve çıkarıldı!")
# Gerekli kütüphaneleri ekleyelim
import os
import torch
from torchvision import datasets, transforms

# Veri seti yolu
dataset_path = "covid19_dataset"

# Eğer veri seti klasörü yoksa, hata vermesin diye kontrol edelim
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Veri seti klasörü bulunamadı: {dataset_path}")

# Veri dönüşümlerini tanımlayalım
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Veri setini yükleyelim
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Eğitim ve test veri kümelerini ayıralım
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Klasör isimlerini etiket olarak alalım
class_names = dataset.classes

# Bilgi yazdıralım
print(f"Veri setindeki sınıflar: {class_names}")
print(f"Eğitim veri sayısı: {len(train_dataset)}, Test veri sayısı: {len(test_dataset)}")
