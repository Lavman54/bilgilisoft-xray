import streamlit as st
from PIL import Image

# CSS ile Dark Mode (Siyah Arka Plan ve Açık Mavi Yazılar)
st.markdown(
    """
    <style>
    /* Sayfanın arka planını siyah yap */
    body {
        background-color: black;
        color: #ADD8E6; /* Açık Mavi */
    }

    /* Başlıkları açık mavi yap */
    h1, h2, h3, h4, h5, h6 {
        color: #ADD8E6; /* Açık Mavi */
    }

    /* Butonları özelleştirme */
    .stButton>button {
        background-color: white;
        color: black;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }

    /* Yüklenen dosya seçme butonu */
    .stFileUploader label {
        color: #ADD8E6; /* Açık Mavi */
    }

    /* Sidebar arka planını koyulaştır */
    .css-1d391kg {
        background-color: #333 !important;
    }

    /* Logo ortalama ve sayfanın altına yerleştirme */
    .footer {
        position: fixed;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
    }
    </style>
    """,
    unsafe_allow_html=True
)

import sys
import os

# Dosya yolunu düzelt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Şimdi import et
from prepare_dataset import class_names

import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from prepare_dataset import class_names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("resnet50_chest_xray.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
st.markdown("<h1 style='text-align: center; color: blue;'>BİLGİLİSOFT</h1>", unsafe_allow_html=True)

# Modeli yükleyelim
from prepare_dataset import class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)  # Eğitilmiş modelimizi çağırıyoruz
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("resnet50_chest_xray.pth", map_location=device))
model.to(device)
model.eval()

# Görüntüyü modele uygun hale getiren dönüşüm fonksiyonları
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit arayüzü
st.title("Akciğer Grafisi Tahmin Sistemi")
st.write("Lütfen bir akciğer grafisi yükleyin.")

# Kullanıcıdan görüntü yüklemesini iste
uploaded_file = st.file_uploader("Görüntü yükleyin", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Görüntüyü ekrana göster
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Akciğer Grafisi", use_column_width=True)

    # Görüntüyü modele uygun hale getir
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Model tahmini yap
    with torch.no_grad():
        output = model(input_tensor)

    # Tahmini al
    _, predicted_class = output.max(1)
    predicted_label = class_names[predicted_class.item()]

    # Sonucu ekrana yazdır
    st.success(f"Tahmin edilen hastalık: {predicted_label}")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 14px;
        color: gray;
    }
    </style>
    <div class="footer">Written by Arda Bilgili</div>
    """,
    unsafe_allow_html=True
)
