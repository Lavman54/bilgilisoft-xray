import os
import matplotlib.pyplot as plt
from PIL import Image

# Veri seti yolu
dataset_path = "C:/Users/ARDA/PycharmProjects/PythonProject1/Proje_Klasörün/covid19_dataset"

# Sınıf klasörleri
categories = ["COVID-19", "NORMAL", "LUNG_OPACITY", "VIRAL_PNEUMONIA"]

# Her sınıftan bir örnek görüntü gösterelim
for category in categories:
    folder_path = os.path.join(dataset_path, category, "images")  # "images" klasörüne eriş

    # Klasörün var olup olmadığını kontrol et
    if not os.path.exists(folder_path):
        print(f"HATA: {folder_path} klasörü bulunamadı!")
        continue

    # Klasör boşsa hata verme, atla
    if not os.listdir(folder_path):
        print(f"HATA: {folder_path} klasörü boş! Lütfen veri setini kontrol et.")
        continue

    image_name = os.listdir(folder_path)[0]  # İlk görüntüyü seç
    image_path = os.path.join(folder_path, image_name)

    # Görüntünün var olup olmadığını kontrol et
    if not os.path.exists(image_path):
        print(f"HATA: {image_path} dosyası bulunamadı!")
        continue

    try:
        # Görüntüyü Pillow (PIL) ile yükle
        image = Image.open(image_path).convert("L")  # Gri tonlamaya çevir
        plt.imshow(image, cmap="gray")
        plt.title(f"Kategori: {category}")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"HATA: {image_path} yüklenemedi! Hata: {e}")

