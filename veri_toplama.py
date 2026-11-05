# Sistemin 1. Adımı: Veri Toplama (FR-1)
# Bu modül, Kaggle'dan alınan tweetleri ve yfinance üzerinden borsa fiyatlarını çeker.

import yfinance as yf
import pandas as pd

def borsa_verisi_cek(hisse_kodu, baslangic, bitis):
    print(f"{hisse_kodu} için borsa verileri çekiliyor...")
    veri = yf.download(hisse_kodu, start=baslangic, end=bitis)
    return veri

def tweet_verisi_yukle(dosya_yolu):
    print("Tweet verileri yükleniyor...")
    df = pd.read_csv(dosya_yolu)
    return df

if __name__ == "__main__":
    # Örnek kullanım
    borsa_df = borsa_verisi_cek("THYAO.IS", "2023-01-01", "2023-06-01")
    print(borsa_df.head())
