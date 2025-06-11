"""
Asetilkolinesteraz (AChE) Enzimi İnhibitörlerinin Makine Öğrenmesi ile Tahmin Edilmesi
Bu proje, Alzheimer gibi nörodejeneratif hastalıkların tedavisinde potansiyel ilaçlar geliştirmeyi hedefleyen bir makine öğrenmesi modelinin tasarımını 
içermektedir. Projenin ana amacı, Asetilkolinesteraz (AChE) inhibitörlerinin etkinliğini tahmin etmek için makine öğrenmesi algoritmalarını kullanmaktır. 
Bu tür inhibitörler, AChE enziminin aktivitesini engelleyerek nörotransmitter asetilkolinin seviyelerini artırır, bu da nörolojik bozuklukların tedavisinde 
önemli bir etkiye sahiptir.Proje kapsamında, büyük veri setlerinden elde edilen biyolojik ve kimyasal özellikler kullanılacaktır. Kimyasal veriler, 
SMILES notasyonu ile ifade edilen moleküller üzerinden elde edilir ve bu veriler, biyolojik aktiviteyi daha iyi temsil edebilmek için uygun bir formatta dönüştürülür. Veri hazırlama aşamasında, çeşitli moleküler özellikler ve parmak izi yöntemleri ile kimyasal ve biyolojik bilgilerin entegrasyonu sağlanacaktır.
Makine öğrenmesi algoritmaları, örneğin Derin Öğrenme, Rastgele Ormanlar (Random Forest) ve Destek Vektör Makineleri (SVM), 
AChE inhibitörlerinin aktif veya inaktif olup olmadığını sınıflandırmak için kullanılacaktır. Modelin başarısı, doğru sınıflandırma sonuçları ve 
yüksek tahmin doğruluğu ile ölçülecektir.Veri hazırlama sürecinin ilk adımı, farklı kaynaklardan elde edilen moleküler veri setlerinin uygun bir 
biçime dönüştürülmesi ve tutarlı hale getirilmesidir. Bu aşama, proje için %10'luk bir katkı sağlar ve hedeflenen başarı oranı en az %95 doğruluk 
ile verilerin uygun formata dönüştürülmesidir.Bu proje, Alzheimer tedavisi için yeni potansiyel ilaçların tasarımını destekleyecek, makine öğrenmesi
ve biyoinformatik alanlarında önemli bir katkı sunmayı amaçlamaktadır.
"""

#İLK AŞAMA:
#Moleküler Veri Setlerinin Hazırlanması: AChE inhibitörlerini tahmin etmek için kullanılacak geniş moleküler veri setlerinin toplanması 
#ve tüm veri setlerinin ortak bir formata dönüştürülmesi. Moleküler veriler arasında tutarlılığı sağlamak ve tek tip bir giriş verisi oluşturmak.

import pandas as pd

df = pd.read_csv("C:\\Users\\nisaa\\OneDrive\\Veri Bilimi ve Makine Ogrenimi\\Projem\\AChE_Bioactivity_data_1.csv")

df.columns = df.columns.str.strip()

print(df.head())
print(df.columns)

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# Kategorik veriler için eksik verileri mod ile doldurmak
categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print(df.head())

# Tüm eksik verileri tekrar kontrol 
print(df.isna().sum())

#2.AŞAMA:
#Özellik Çıkarımı (Feature Extraction)
#Bu aşamada, molekülleri daha iyi temsil edebilmek için kimyasal ve biyolojik özellikler çıkarılacak. Özellik çıkarımı, makine öğrenmesi modellerine
#girdi olacak verileri oluşturmak için gereklidir. RDKit gibi kütüphaneler kullanarak, moleküllerin çeşitli kimyasal özelliklerini ve 
#parmak izi yöntemlerini çıkarabiliriz

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

# SMILES notasyonlarını içeren sütun
smiles_column = 'canonical_smiles'

def extract_molecule_features(smiles):
    try:
        # SMILES notasyonundan molekül objesi oluştur
        molecule = Chem.MolFromSmiles(smiles)
        
        # Molekülün özelliklerini çıkaralım
        if molecule:
            mol_weight = Descriptors.MolWt(molecule)  # Moleküler Ağırlık
            logp = Descriptors.MolLogP(molecule)  # LogP (Yağda çözünürlük)
            hba = Descriptors.NumHAcceptors(molecule)  # Hidrojen Bağlayıcı Alıcı Sayısı
            hbd = Descriptors.NumHDonors(molecule)  # Hidrojen Bağlayıcı Donör Sayısı
            
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
            
            return mol_weight, logp, hba, hbd, fingerprint
        else:
            return None, None, None, None, None
    except:
        return None, None, None, None, None

# SMILES notasyonları üzerinden özellik çıkarımını yapalım
df[['MolWt', 'LogP', 'HBA', 'HBD', 'Fingerprint']] = df[smiles_column].apply(lambda x: pd.Series(extract_molecule_features(x)))

# İlk 5 satırı görelim
print(df[['canonical_smiles', 'MolWt', 'LogP', 'HBA', 'HBD']].head())

# Yeni oluşturulan özellikler ve parmak izlerini veri setine ekledik.
#parmakizi çıktısı
import numpy as np

def extract_molecule_features(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            mol_weight = Descriptors.MolWt(molecule)
            logp = Descriptors.MolLogP(molecule)
            hba = Descriptors.NumHAcceptors(molecule)
            hbd = Descriptors.NumHDonors(molecule)

            # Morgan parmak izini çıkar ve listeye çevir
            fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
            fingerprint_list = list(fingerprint)  # Liste formatına çevir

            # Listeyi string formatına çevir
            fingerprint_str = ','.join(map(str, fingerprint_list))

            return mol_weight, logp, hba, hbd, fingerprint_str  # Parmak izini string olarak döndür
        else:
            return None, None, None, None, None
    except Exception as e:
        print(f"Hata: {e}")
        return None, None, None, None, None

# Özellik çıkarımını uygula
df[['MolWt', 'LogP', 'HBA', 'HBD', 'Fingerprint']] = df[smiles_column].apply(lambda x: pd.Series(extract_molecule_features(x)))

# İlk 5 satırı kontrol et
print(df[['canonical_smiles', 'MolWt', 'LogP', 'HBA', 'HBD', 'Fingerprint']].head())
