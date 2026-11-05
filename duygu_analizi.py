# Sistemin 2. Adımı: Büyük Dil Modeli (FinBERT) Duygu Analizi (FR-2)
# Bu modül, tweet metinlerini alıp -1 ile +1 arasında duygu skoru üretir.

from transformers import pipeline

def model_yukle():
    print("FinBERT modeli yükleniyor...")
    # HuggingFace üzerinden FinBERT duygu analizi modeli yükleniyor
    nlp_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return nlp_model

def duygu_skoru_hesapla(metinler):
    model = model_yukle()
    sonuclar = model(metinler)
    # Gelen sonuçları -1 (Negatif) ve +1 (Pozitif) arasına ölçekliyoruz
    skorlar = []
    for sonuc in sonuclar:
        if sonuc['label'] == 'positive':
            skorlar.append(sonuc['score'])
        elif sonuc['label'] == 'negative':
            skorlar.append(-sonuc['score'])
        else:
            skorlar.append(0.0)
    return skorlar

if __name__ == "__main__":
    ornek_tweetler = [
        "Şirket kârını beklenenin üzerinde açıkladı, hisseler yükselecek!",
        "Piyasada sert düşüş var, yatırımcılar endişeli."
    ]
    skorlar = duygu_skoru_hesapla(ornek_tweetler)
    print("Tweet Skorları:", skorlar)
