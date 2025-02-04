import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from prepare_dataset import train_loader, test_loader, \
    class_names  # Daha önce oluşturduğumuz dataset dosyasından veri yükle

# Cihazı kontrol edelim (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# ResNet-50 modelini yükleyelim
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.to(device)

# Modelin son katmanını değiştirelim (Çıkış katmanı 4 sınıf olacak)
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Kaybı (loss function) ve optimizasyonu tanımlayalım
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Eğitim döngüsü parametreleri
num_epochs = 3  # CPU olduğu için ilk deneme için epoch sayısını azaltalım


# Modeli eğitme fonksiyonu
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # Modeli eğitim moduna al
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # GPU veya CPU'ya taşı
            optimizer.zero_grad()  # Gradyanları sıfırla
            outputs = model(images)  # Modeli çalıştır
            loss = criterion(outputs, labels)  # Kayıp fonksiyonunu hesapla
            loss.backward()  # Geri yayılım (backpropagation)
            optimizer.step()  # Model ağırlıklarını güncelle

            running_loss += loss.item()
            if batch_idx % 10 == 0:  # Her 10 batch'te bir loss değerini yazdır
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}")
# Modeli eğit
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Modeli kaydet
torch.save(model.state_dict(), "C:/Users/ARDA/PycharmProjects/PythonProject1/Proje_Klasörün/resnet50_chest_xray.pth")
print("Model başarıyla kaydedildi!")
