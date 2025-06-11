# Prediction-of-AChE-Enzyme-Inhibitors-Using-Machine-Learning
Term Project
# AChE İnhibitörü Biyolojik Aktivite Tahmini

Bu proje, Asetilkolinesteraz (AChE) inhibitörlerinin biyolojik aktivitesini tahmin etmek için makine öğrenmesi yöntemlerini kullanır. AChE inhibitörleri, Alzheimer gibi nörodejeneratif hastalıkların tedavisinde önemli bir rol oynar. Projede, moleküler yapıların SMILES temsillerinden kimyasal ve biyolojik özellikler çıkarılarak sınıflandırma yapılmıştır.

## 🚀 Proje İçeriği

- RDKit ile moleküler özellik çıkarımı (MolWt, LogP, TPSA, vb.)
- Morgan Fingerprint (1024-bit)
- Özellik standardizasyonu ve veri ön işleme
- XGBoost sınıflandırma modeli
- SMOTE ile veri dengeleme
- Eğitim ve tahmin sonrası istatistiksel değerlendirme (confusion matrix, accuracy, precision, recall, f1-score)
- Kullanıcıdan SMILES alarak tahmin yapan fonksiyon

## 📁 Klasör ve Dosya Açıklamaları

- `proje.py`: Veri temizleme, özellik çıkarımı ve model eğitimi
- `predict.py`: Yeni bir molekül için tahmin yapan betik
- `AChE_Processed.csv`: İşlenmiş veri seti
- `model_xgboost.pkl`: Eğitilmiş XGBoost modeli
- `scaler.pkl`: Özellik ölçekleyici
- `README.md`: Proje açıklaması

## ⚙️ Gereksinimler

- Python 3.8+
- RDKit
- Scikit-learn
- XGBoost
- Pandas, NumPy, Matplotlib

Kurulum için:
```bash
pip install -r requirements.txt

