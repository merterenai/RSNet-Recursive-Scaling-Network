import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import urllib.request
from rs_net import RSNet

class CryptoPredictorRSNet(nn.Module):
    """
    Wrapper class adapting the RSNet architecture for cryptocurrency price/return prediction.
    RSNet mimarisini kripto para fiyat/getiri tahmini için uyarlayan sarmalayıcı sınıf.
    
    It maps the final RSNet node representation to a single scalar value.
    Nihai RSNet düğüm temsilini tek bir skaler değere (tahmine) dönüştürür.
    """
    def __init__(self, seq_len, dim):
        super().__init__()
        self.rsnet = RSNet(num_inputs=seq_len, input_dim=dim)
        self.fc = nn.Linear(dim, 1) 
        
    def forward(self, x_list):
        rs_out = self.rsnet(x_list)
        return self.fc(rs_out)

class CryptoPredictorRNN(nn.Module):
    """
    Wrapper class for standard Recurrent Neural Networks (RNN, LSTM, GRU).
    Standart Tekrarlayan Sinir Ağları (RNN, LSTM, GRU) için sarmalayıcı sınıf.
    
    Serves as the baseline for evaluating RSNet's performance.
    RSNet'in performansını değerlendirmek için temel (baseline) görevi görür.
    """
    def __init__(self, dim, hidden_dim, rnn_type='LSTM'):
        super().__init__()
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(dim, hidden_dim, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(dim, hidden_dim, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(dim, hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)

def fetch_real_crypto_data(symbol="BTCUSDT", seq_len=8, total_points=5000):
    """
    Fetches real historical OHLCV data from the Binance Public API via pagination.
    This ensures benchmarks are conducted on real-world, non-stationary financial data.
    
    Binance Genel API'sinden sayfalama (pagination) yoluyla gerçek geçmiş OHLCV verilerini çeker.
    Bu, testlerin gerçek dünyadaki (durağan olmayan) finansal veriler üzerinde yapılmasını sağlar.
    """
    ohlcv = []
    end_time = None
    limit = 1000 # Binance API maximum limit per request / Binance API istek başına maksimum limit
    
    print(f"Downloading {total_points} hourly (1h) records for {symbol} from Binance...")
    print(f"Binance'den {symbol} için {total_points} saatlik (1s) kayıt indiriliyor...")
    
    while len(ohlcv) < total_points:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit={limit}"
        if end_time:
            url += f"&endTime={end_time}"
            
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        if not data:
            break
            
        chunk = []
        for row in data:
            chunk.append([
                float(row[1]), # Open / Açılış
                float(row[2]), # High / En Yüksek
                float(row[3]), # Low / En Düşük
                float(row[4]), # Close / Kapanış
                float(row[5])  # Volume / Hacim
            ])
            
        ohlcv = chunk + ohlcv
        end_time = data[0][0] - 1
        time.sleep(0.1) # Rate limiting protection / API limit koruması
        
    ohlcv = torch.tensor(ohlcv[-total_points:], dtype=torch.float32)
    
    X_list = []
    Y_list = []
    
    # Windowing for Time-Series Modeling
    # Zaman Serisi Modellemesi için Pencereleme
    for i in range(len(ohlcv) - seq_len):
        window = ohlcv[i : i + seq_len]
        
        # MITIGATING NAIVE FORECAST:
        # Instead of predicting absolute price, we predict the percentage return.
        # This prevents the model from artificially minimizing loss by just repeating the last price.
        
        # NAİF TAHMİNİ (NAIVE FORECAST) ÖNLEME:
        # Mutlak fiyatı tahmin etmek yerine, yüzdelik getiriyi (değişimi) tahmin ediyoruz.
        # Bu, modelin sadece son fiyatı tekrarlayarak kaybı (loss) yapay olarak düşürmesini engeller.
        last_close = window[-1, 3]
        next_close = ohlcv[i + seq_len, 3]
        target_return = ((next_close - last_close) / last_close) * 100.0 # Percentage change / Yüzdelik değişim
        
        # PER-WINDOW NORMALIZATION:
        # Prevents data leakage and ensures the model evaluates local volatility patterns.
        
        # PENCERE BAZLI NORMALİZASYON:
        # Veri sızıntısını (data leakage) önler ve modelin yerel volatilite modellerini değerlendirmesini sağlar.
        w_mean = window.mean(dim=0, keepdim=True)
        w_std = window.std(dim=0, keepdim=True) + 1e-8
        window_norm = (window - w_mean) / w_std
        
        X_list.append(window_norm)
        Y_list.append(torch.tensor([target_return], dtype=torch.float32))
        
    X = torch.stack(X_list)
    Y = torch.stack(Y_list)
    
    # Split into Train (80%) and Validation (20%) / Eğitim (%80) ve Doğrulama (%20)
    split_idx = int(len(X) * 0.8)
    train_X, val_X = X[:split_idx], X[split_idx:]
    train_Y, val_Y = Y[:split_idx], Y[split_idx:]
    
    print(f"Train Dataset Shape (Eğitim Verisi Boyutu): X={train_X.shape}, Y={train_Y.shape}")
    print(f"Validation Dataset Shape (Doğrulama Verisi Boyutu): X={val_X.shape}, Y={val_Y.shape}")
    
    return train_X, train_Y, val_X, val_Y

def train_and_eval(model, model_name, train_X, train_Y, val_X, val_Y, epochs=100, lr=0.001):
    """
    Standard training and evaluation loop. Records Mean Squared Error (MSE) and Directional Accuracy.
    Standart eğitim ve değerlendirme döngüsü. Ortalama Kare Hata (MSE) ve Yönsel Doğruluğu (Directional Accuracy) kaydeder.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Format inputs based on architecture type (List for RSNet, Tensor for RNNs)
    # Mimari türüne göre girdileri biçimlendir (RSNet için Liste, RNN'ler için Tensör)
    if model_name == 'RSNet':
        inputs_train = [train_X[:, i, :] for i in range(train_X.size(1))]
        inputs_val = [val_X[:, i, :] for i in range(val_X.size(1))]
    else:
        inputs_train = train_X
        inputs_val = val_X
        
    start_train = time.time()
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs_train)
        loss = criterion(outputs, train_Y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        # Gradyan patlamasını önlemek için gradyan kırpma
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    train_time = time.time() - start_train
    final_train_loss = loss.item()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(inputs_val)
        val_loss = criterion(val_outputs, val_Y).item()
        
        # Directional Accuracy Calculation
        # Checks if the model correctly predicted the upward or downward trend
        
        # Yönsel Doğruluk (Directional Accuracy) Hesaplaması
        # Modelin yukarı veya aşağı trendi (yönü) doğru tahmin edip etmediğini kontrol eder
        pred_direction = (val_outputs > 0).float()
        true_direction = (val_Y > 0).float()
        directional_acc = (pred_direction == true_direction).float().mean().item() * 100.0
        
    return final_train_loss, val_loss, train_time, directional_acc

def run_crypto_benchmark():
    seq_len = 8   
    dim = 5       
    
    train_X, train_Y, val_X, val_Y = fetch_real_crypto_data(symbol="BTCUSDT", seq_len=seq_len, total_points=50000)
    
    models = {
        'RSNet': CryptoPredictorRSNet(seq_len, dim),
        'RNN': CryptoPredictorRNN(dim, hidden_dim=32, rnn_type='RNN'),
        'GRU': CryptoPredictorRNN(dim, hidden_dim=32, rnn_type='GRU'),
        'LSTM': CryptoPredictorRNN(dim, hidden_dim=32, rnn_type='LSTM')
    }
    
    print(f"\n{'Model':<10} | {'Train Loss(MSE)':<15} | {'Val Loss(MSE)':<15} | {'Directional Acc':<15} | {'Train Time'}")
    print("-" * 85)
    
    for name, model in models.items():
        t_loss, v_loss, t_time, dir_acc = train_and_eval(model, name, train_X, train_Y, val_X, val_Y, epochs=150)
        print(f"{name:<10} | {t_loss:<15.4f} | {v_loss:<15.4f} | {dir_acc:<14.2f}% | {t_time:.2f} s")

if __name__ == '__main__':
    run_crypto_benchmark()
