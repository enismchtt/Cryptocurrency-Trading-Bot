import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CSV oku
df = pd.read_csv("latest_dataset/BTC/BTCUSDT_1d.csv")

# Sadece numeric kolonlar
numeric_df = df.select_dtypes(include="number")

# Korelasyon matrisi
corr_matrix = numeric_df.corr()

# Çıktı klasörü
output_dir = Path("correlation_outputs")
output_dir.mkdir(exist_ok=True)

# Plot - GÜNCELLEME: cmap='coolwarm', vmin ve vmax eklendi
plt.figure(figsize=(10, 8))
im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im)
plt.title("Correlation Matrix (Coolwarm)")

# Eksen etiketleri
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

# Değerleri yaz
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=8,
                 color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black") 


output_path = output_dir / "correlation_matrix_coolwarm.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Grafik 'coolwarm' temasıyla kaydedildi: {output_path}")