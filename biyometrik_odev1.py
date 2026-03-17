import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Veriyi yükleme
data = np.load("Features.npz")
Features = data["Features"]

# Sadece ilk 100 kişinin verisini kullan
Features = Features[:, :100, :]

# Normalizasyon işlemi (0-1 aralığı)
min_val = Features.min()
max_val = Features.max()
Features = (Features - min_val) / (max_val - min_val)

genuine_scores = []
imposter_scores = []

sessions = Features.shape[0]
persons = Features.shape[1]

# Genuine skorlarının hesaplanması (aynı kişinin farklı oturumları)
for p in range(persons):
    for i in range(sessions):
        for j in range(i+1, sessions):
            d = euclidean(Features[i,p], Features[j,p])
            score = 1/(1+d)
            genuine_scores.append(score)

# Imposter skorlarının hesaplanması (farklı kişiler)
for i in range(sessions):
    for p1 in range(persons):
        for p2 in range(p1+1, persons):
            d = euclidean(Features[i,p1], Features[i,p2])
            score = 1/(1+d)
            imposter_scores.append(score)

genuine_scores = np.array(genuine_scores)
imposter_scores = np.array(imposter_scores)

# Skor dağılımlarının çizdirilmesi
plt.figure(figsize=(8,5))
plt.hist(genuine_scores, bins=40, alpha=0.6, label="Gerçek (Genuine)")
plt.hist(imposter_scores, bins=40, alpha=0.6, label="Sahteci (Imposter)")

plt.xlabel("Skor Değeri")
plt.ylabel("Frekans")
plt.title("Skor Dağılımları")
plt.legend()

# Ortalama değerleri grafikte göster
plt.text(0.02, plt.ylim()[1]*0.9,
         f"Genuine Ortalama: {np.mean(genuine_scores):.3f}\nImposter Ortalama: {np.mean(imposter_scores):.3f}",
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.show()

# Threshold değerleri
thresholds = np.linspace(0,1,500)

FAR = []
FRR = []

# FAR ve FRR hesaplama
for t in thresholds:

    far = np.sum(imposter_scores >= t) / len(imposter_scores)
    frr = np.sum(genuine_scores < t) / len(genuine_scores)

    FAR.append(far)
    FRR.append(frr)

FAR = np.array(FAR)
FRR = np.array(FRR)

# EER hesaplama
idx = np.argmin(np.abs(FAR-FRR))
EER = FAR[idx]
eer_threshold = thresholds[idx]
EER_percent = EER * 100

print(f"Eşit Hata Oranı (EER): %{EER_percent:.2f}")
print(f"EER için eşik değeri: {eer_threshold:.4f}")

# FAR ve FRR grafiği
plt.figure(figsize=(8,5))

plt.plot(thresholds, FAR, label="FAR (Yanlış Kabul Oranı)")
plt.plot(thresholds, FRR, label="FRR (Yanlış Ret Oranı)")

plt.axvline(eer_threshold, linestyle="--", label=f"EER = %{EER_percent:.2f}")

plt.xlabel("Eşik Değeri (Threshold)")
plt.ylabel("Hata Oranı")
plt.title("FAR ve FRR Değişimi")

# EER noktasını grafikte göster
plt.scatter(eer_threshold, EER, zorder=5)

plt.text(eer_threshold, EER,
         f"  EER: %{EER_percent:.2f}\n  Threshold: {eer_threshold:.3f}",
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.legend()
plt.grid(alpha=0.3)

plt.show()

# FAR vs FRR grafiği
plt.figure(figsize=(6,6))

plt.plot(FAR, FRR, label="FAR - FRR Eğrisi")

plt.scatter(EER, EER, zorder=5)

plt.text(EER, EER,
         f"EER = %{EER_percent:.2f}",
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel("FAR (Yanlış Kabul Oranı)")
plt.ylabel("FRR (Yanlış Ret Oranı)")
plt.title("FAR vs FRR Eğrisi")

plt.legend()
plt.grid(alpha=0.3)

plt.show()