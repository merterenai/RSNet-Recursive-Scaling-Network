# RSNet (Recursive Scaling Network)

RSNet, dizi modelleme (sequence modeling) problemleri için tasarlanmış hiyerarşik ve ağaç (tree) tabanlı yepyeni bir sinir ağı mimarisidir. 

Geleneksel Recurrent Neural Network (RNN, LSTM, GRU) türevlerinin sahip olduğu **O(N) ardışık işlem darboğazını** çözmek amacıyla geliştirilmiştir. RSNet, diziyi baştan sona ardışık olarak işlemek yerine her derinlik seviyesinde yan yana olan elemanları kendi özgün çarpımsal formülü ile eşleştirerek diziyi logaritmik olarak küçültür. 

Bu eşsiz yaklaşım sayesinde RSNet'in zaman karmaşıklığı (Time Complexity), dizinin uzunluğu $N$ iken **$O(\log_2(N))$** olmaktadır.

## RSNet'in Matematiksel Yapısı

Mimari, `RSBlock` adı verilen temel hesaplama birimlerinden oluşur. Her blok, kendisine gelen iki alt düğümü (X1, X2) alır ve ağacın bir üst seviyesine şu çarpımsal-oran (multiplicative-ratio) formülü ile aktarır:

1. $Y_1 = X_1 \cdot W_1$
2. $Y_2 = X_2 \cdot W_2$
3. $F_1 = Y_1 \cdot Y_2$ *(Çarpımsal Birleşim)*
4. $F_2 = \frac{F_1}{|Y_1 + Y_2| + \epsilon}$ *(Dinamik Oranlama)*
5. $X_3 = \text{LeakyReLU}(F_2)$
6. $Y_3 = X_3 \cdot W_3$
7. $\text{Çıktı} = \text{LeakyReLU}(Y_3)$

Bu yapı, modelin dizisel verilerdeki karmaşık ve uzun vadeli ilişkileri (long-term dependencies) patlayan gradyan (exploding gradient) sorunu yaşamadan öğrenmesini sağlar.

---

## Kurulum ve Sistem Gereksinimleri

Proje tamamen **saf (pure) PyTorch** kullanılarak geliştirilmiştir. Herhangi bir karmaşık C++ eklentisine, CUDA derleyicisine veya harici Linux kütüphanesine ihtiyaç duymaz. Hatta Mamba benchmark testleri bile Windows uyumluluğu için saf PyTorch ile sıfırdan yazılmıştır (`mamba_minimal.py`).

Yalnızca PyTorch kurarak tüm projeyi çalıştırabilirsiniz:
```bash
pip install torch
```

---

## Benchmark Testleri

Projede, RSNet'in kapasitesini kanıtlamak üzere geleneksel modellerle (RNN, LSTM, GRU ve Mamba) karşılaştıran farklı zorluk seviyelerinde 4 ayrı test ortamı bulunmaktadır:

### 1. Kapsamlı Görev Testi (`benchmark_rsnet.py`)
RSNet'in kapasitesini 10 farklı görevde (Doğrusal Toplam, Sinüs Dönüşümü, Mutlak Değer Karmaşası vb.) test eden temel test ortamıdır. RSNet'in öğrenme yeteneğini izole olarak ölçer.
```bash
python benchmark_rsnet.py
```

### 2. Dizi Uzunluğu (Sequence Length) Ölçekleme Testi (`benchmark_seqlen_all.py`)
Modelin $O(\log N)$ derinlik avantajını ispatlayan ana testtir. RSNet, RNN, GRU, LSTM ve Mamba'yı `seq_len = [4, 8, 16, 32, 64, 128]` şeklinde üssel olarak artan dizi uzunlukları boyunca adil bir sentetik veride değerlendirir. 
```bash
python benchmark_seqlen_all.py
```

### 3. Kripto Para / Gerçek Finansal Veri Testi (`crypto_benchmark.py`)
Binance API'si üzerinden geçmiş saatlik BTCUSDT verilerini sayfalamayla (pagination) indirerek modellerin gerçek dünya (durağan olmayan / non-stationary) verisindeki performansını ölçer. 
*"Naif Tahmin (Naive Forecast)"* hilesini (dünün fiyatını yarına kopyalama) engellemek için, modellerden mutlak fiyat yerine **yüzdelik getiri (%)** tahmini yapmaları istenir.
```bash
python crypto_benchmark.py
```

### 4. Çarpımsal ve Ardışık Fark Hedefleri Testi (`karsilastirmali_benchmark.py`)
RNN'lerin doğası gereği çözmekte aşırı zorlandığı (gradyan patlaması/kaybolması yaratan) hedefleri test eder:
- **Çarpımsal İlişki:** $x[t] \cdot x[t-1] \cdot \dots$
- **Ardışık Farklar:** $f(x[t] - x[t-1])$

RSNet'in ağaç yapısının bu tarz doğrusal olmayan (non-linear) uç hedefleri nasıl başarıyla yönettiğini gösterir.
```bash
python karsilastirmali_benchmark.py
```

---

## Dil ve Dokümantasyon Standardı
Açık kaynak (Open-Source) ekosistemine global boyutta uyum sağlamak ve aynı zamanda yerel araştırmacılara destek olmak amacıyla projedeki tüm kod içi açıklamalar (docstrings ve comments) **İngilizce ve Türkçe (Çift Dilli - Bilingual)** olarak hazırlanmıştır.