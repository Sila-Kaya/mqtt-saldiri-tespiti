# MQTT saldırı tespiti deneyleri

MQTTEEB-D ve MQTTset veri setlerini okur, iki veri setinde ortak olan özellikleri bulur ve üç deneyi tek komutla çalıştırır:

| Deney | Soru | Modeller |
|---|---|---|
| 1. Bilinen saldırılar | MQTTEEB-D'de saldırı tipleri ne kadar iyi ayrılıyor? | Decision Tree, Random Forest |
| 1b. Bölme yöntemi | Rastgele bölme sonucu şişiriyor mu? | DT, RF |
| 1c. Sızıntı testi | IP, port, zaman, client ID gibi sütunlar modele verilirse ne olur? | RF |
| 2. Bilinmeyen saldırılar | Eğitimde hiç görülmeyen saldırı tipi yakalanıyor mu? (Leave-One-Attack-Type-Out) | DT, RF, Isolation Forest |
| 3. Veri setleri arası | MQTTEEB-D ile eğitilen model MQTTset'te çalışıyor mu? (ve tersi) | DT, RF, Isolation Forest |

Sonunda `sonuclar/rapor.md` dosyasına tüm tabloları içeren bir rapor yazar.

## 1. Kurulum

```bash
cd ~/Desktop/mqtt-saldiri-tespiti
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Bilgisayarındaki varsayılan `python3` sürümü 3.15 ve pandas/scikit-learn bu sürüme henüz kurulamayabilir. Bu yüzden `python3.14` kullan; bu sürümle test edildi.

## 2. Veri setlerini indir

**MQTTEEB-D:** [Mendeley Data](https://data.mendeley.com/datasets/jfttfjn6tr/1)
- **Ham (raw) ya da cleaned CSV'yi kullan.**
- SMOTE, normalized ve standardized sürümlerini kullanma. Bu sürümler test sonuçlarını şişirir; kod da bu adları taşıyan dosyaları reddeder.
- Dosyayı `veri/MQTTEEB-D/` klasörüne koy.

**MQTTset:** Kaggle'da "MQTTset" diye ara.
- Örneğin `train70_reduced.csv` ve `test30_reduced.csv` dosyalarını birlikte ver. Kod bunları birleştirip kendisi yeniden böler.
- `augmented` sürümleri kullanma.
- Dosyaları `veri/MQTTset/` klasörüne koy.

## 3. Önce ortak özelliklere bak

```bash
python mqtt_deneyler.py \
  --a veri/MQTTEEB-D/DOSYA_ADI.csv \
  --b veri/MQTTset/train70_reduced.csv veri/MQTTset/test30_reduced.csv \
  --sadece-ozellikler
```

Konsolda iki şeyi kontrol et:

1. **Etiket eşlemesi:** Örneğin `'SlowITe' -> slowite`. Yanlış eşlenen bir etiket varsa koddaki `LABEL_RULES` listesini düzelt.
2. **Ortak özellik sayısı.** Tüm sütunların durumu `sonuclar/ozellik_karsilastirma.csv` dosyasında.

Aynı alan iki veri setinde farklı adla geçiyorsa eşle, örneğin `--esle mqtt_message_type=mqtt_msgtype`. Sağdaki ad MQTTEEB-D'deki ad olmalı.

## 4. Deneyleri çalıştır

Önce hızlı bir deneme yap (sınıf başına 20.000 satır, 50 ağaç):

```bash
python mqtt_deneyler.py --a veri/MQTTEEB-D/DOSYA_ADI.csv \
  --b veri/MQTTset/train70_reduced.csv veri/MQTTset/test30_reduced.csv --hizli
```

Sonra asıl çalıştırmayı yap. Varsayılan ayar sınıf başına 50.000 satır ve 100 ağaçtır; tüm veri için `--sinif-basina-max 0` ver:

```bash
python mqtt_deneyler.py --a veri/MQTTEEB-D/DOSYA_ADI.csv \
  --b veri/MQTTset/train70_reduced.csv veri/MQTTset/test30_reduced.csv
```

Süre makineye ve satır sayısına göre birkaç dakikadan yarım saate kadar değişebilir.

## Çıktılar (`sonuclar/`)

| Dosya | İçerik |
|---|---|
| `rapor.md` | Bütün tabloların toplandığı rapor |
| `ozellik_karsilastirma.csv` | Her sütun hangi veri setinde var, kullanıldı mı, neden atıldı |
| `teshis.csv` | Tekrar eden ve çelişkili satır oranları; test satırlarının eğitimde birebir bulunma oranı |
| `deney1_ozet.csv`, `deney1_sinif_bazinda.csv` | Bilinen saldırı sonuçları (blok ve rastgele bölme, sızıntı testi) |
| `deney1_karisiklik_*.png` | Karışıklık matrisleri |
| `deney1_ozellik_onemi.csv` | Random Forest özellik önemleri |
| `deney1c_sizinti_ozellik_onemi.csv` | Sızıntı sütunları eklenince modelin neye baktığı |
| `deney2_ana_tablo_yakalama.csv` | **Ana tablo:** dışarıda bırakılan saldırı × model, yakalama oranı |
| `deney2_yanlis_alarm.csv`, `deney2_loao.csv` | Yanlış alarm, F1, ROC-AUC, PR-AUC |
| `deney2_yakalama.png` | Ana tablonun grafiği |
| `deney3_ozet.csv`, `deney3_*_bazinda.csv` | Alan içi ve alanlar arası sonuçlar |
| `deney3_genelleme.png`, `deney3_karisiklik_*.png` | Genelleme grafiği ve karışıklık matrisleri |

## Yöntem notları (hocaya anlatırken)

- **Blok bölme:** Her sınıfın zamanca ilk %70'i eğitime, son %30'u teste ayrılır. Zaman sütunu varsa satırlar önce zamana göre sıralanır. Ardışık paketler hem eğitime hem teste düşmediği için ezber azalır. Rastgele bölmeyle fark Deney 1b'de görülür.
- **Sızıntı sütunları atılır:** IP, port, MAC, zaman damgası, frame numarası, `tcp.stream`, TCP sıra numaraları, TTL, client ID ve mesaj ID. Liste koddaki `LEAK_PATTERNS` içinde; `--ekstra-sizinti` ile genişletebilir, `--tut` ile tek tek geri alabilirsin.
- **Metin sütunları atılır:** Topic adı, payload ve protokol adı gibi sayıya çevrilemeyen sütunlar. Hex değerler (`0x18`) ve tshark çoklu değerleri (`3,3`) sayıya çevrilir. Pakette bulunmayan alanlar -1 olur.
- **LOAO:** Her katlamada bir saldırı tipi eğitimden tamamen çıkarılır ve problem ikili sınıflandırmaya döner (saldırı mı, normal mi). Ölçülen şey, dışarıda kalan saldırının ne oranda "saldırı" olarak işaretlendiği.
- **Isolation Forest:** Yalnızca normal trafikle eğitilir. Eşik, ayrılmış normal trafikte %1 yanlış alarm verecek şekilde seçilir (`--hedef-fpr`); saldırı etiketlerine hiç bakılmaz. Deney 3'te bu eşik olduğu gibi diğer veri setine taşınır. Eşiğin yeni ağda kayması da bir bulgu.
- **Veri setleri arası test:** Yalnızca iki veri setinde ortak olan özellikler ve sınıflar kullanılır. Karşılaştırma aynı test kümesi üzerinde yapılır: "alan içi" (hedef veri setiyle eğitilmiş) ve "alanlar arası" (diğer veri setiyle eğitilmiş).
- **Sınırlılıklar:**
  - Veriler paket düzeyinde. Tek bir flood paketi normal bir paketten farksız olabilir.
  - Etiket eşlemesi anahtar kelimeyle yapılıyor; çıktıyı kontrol et.
  - Sınıf başına örnekleme (`--sinif-basina-max`) sonuçları biraz değiştirebilir. Farklı `--tohum` değerleriyle tekrarla.

## Tüm seçenekler

```bash
python mqtt_deneyler.py --help
```
