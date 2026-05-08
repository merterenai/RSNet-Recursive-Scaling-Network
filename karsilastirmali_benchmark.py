import torch
import torch.nn as nn
import torch.optim as optim
import time
from rs_net import RSNet
from mamba_minimal import CryptoPredictorMamba

class CryptoPredictorRSNet(nn.Module):
    """
    Wrapper class adapting RSNet for sequence-to-scalar prediction tasks.
    RSNet mimarisini dizi-skaler tahmini görevleri için uyarlayan sarmalayıcı sınıf.
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

def generate_raw_synthetic_data(total_points=10000, dim=5):
    """
    Generates the base synthetic time-series data.
    Temel sentetik zaman serisi verisini üretir.
    """
    t = torch.linspace(0, 100, total_points)
    raw_data = torch.zeros(total_points, dim)
    for d in range(dim):
        signal = torch.sin(t * (d + 1) * 0.5) + torch.randn(total_points) * 0.1
        signal += torch.cumsum(torch.randn(total_points) * 0.05, dim=0)
        raw_data[:, d] = signal
    return raw_data

def prepare_synthetic_data_with_custom_targets(raw_data, target_type='multiplicative'):
    """
    Slices the raw data and computes highly non-linear specific targets to benchmark
    model capacity against extreme temporal relationships. Sequence length is fixed at 8.
    
    Ham veriyi böler ve modelin kapasitesini aşırı (zorlu) zamansal ilişkilere karşı test etmek için
    doğrusal olmayan özel hedefler hesaplar. Dizi uzunluğu 8 olarak sabitlenmiştir.
    """
    seq_len = 8
    X_list = []
    Y_list = []
    
    for i in range(len(raw_data) - seq_len):
        window = raw_data[i : i + seq_len]
        
        # Per-window normalization to prevent exploding inputs
        # Girdilerin patlamasını önlemek için pencere bazlı normalizasyon
        w_mean = window.mean(dim=0, keepdim=True)
        w_std = window.std(dim=0, keepdim=True) + 1e-8
        window_norm = (window - w_mean) / w_std
        
        if target_type == 'multiplicative':
            # Multiplicative Target: x[t] * x[t-1] * ... 
            # Extremely hard for RNNs due to vanishing/exploding gradients. RSNet's tree structure handles this better.
            
            # Çarpımsal Hedef: x[t] * x[t-1] * ... 
            # Gradyan kaybolması/patlaması nedeniyle RNN'ler için son derece zordur. RSNet'in ağaç yapısı bunu daha iyi yönetir.
            target = torch.prod(window_norm[:, 0])
            
        elif target_type == 'comparative':
            # Comparative Target: Sum of squared differences between consecutive elements.
            # Models volatility and temporal jumps.
            
            # Karşılaştırmalı Hedef: Ardışık elemanlar arasındaki karesel farkların toplamı.
            # Volatiliteyi ve zamansal sıçramaları (jumps) modeller.
            diffs = window_norm[1:, 0] - window_norm[:-1, 0]
            target = torch.sum(diffs**2) - torch.mean(diffs)
            
        X_list.append(window_norm)
        Y_list.append(torch.tensor([target], dtype=torch.float32))
        
    X = torch.stack(X_list)
    Y = torch.stack(Y_list)
    
    # Global Z-Score normalization for the targets to stabilize the MSE loss scale
    # MSE kayıp (loss) ölçeğini stabilize etmek için hedeflere (targets) genel Z-Score normalizasyonu uygulanır
    Y_mean = Y.mean()
    Y_std = Y.std() + 1e-8
    Y = (Y - Y_mean) / Y_std
    
    # 80/20 Train-Test Split / Eğitim (%80) ve Doğrulama (%20) Bölünmesi
    split_idx = int(len(X) * 0.8)
    train_X, val_X = X[:split_idx], X[split_idx:]
    train_Y, val_Y = Y[:split_idx], Y[split_idx:]
    
    return train_X, train_Y, val_X, val_Y

def train_and_eval(model, model_name, train_X, train_Y, val_X, val_Y, epochs=150, lr=0.005):
    """
    Executes training and evaluation, recording computation metrics.
    Eğitim ve değerlendirmeyi yürütür, hesaplama metriklerini kaydeder.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if model_name == 'RSNet':
        inputs_train = [train_X[:, i, :] for i in range(train_X.size(1))]
        inputs_val = [val_X[:, i, :] for i in range(val_X.size(1))]
    else:
        inputs_train = train_X
        inputs_val = val_X
        
    # --- Inference Time Measurement / Çıkarım Süresi Ölçümü ---
    model.eval()
    with torch.no_grad():
        _ = model(inputs_val) 
        
        start_inf = time.perf_counter()
        for _ in range(10): 
            _ = model(inputs_val)
        inf_time_ms = ((time.perf_counter() - start_inf) / 10.0) * 1000

    # --- Training Phase / Eğitim Aşaması ---
    start_train = time.time()
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs_train)
        loss = criterion(outputs, train_Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    train_time = time.time() - start_train
    final_train_loss = loss.item()
    
    # --- Validation Phase / Doğrulama Aşaması ---
    model.eval()
    with torch.no_grad():
        val_outputs = model(inputs_val)
        val_loss = criterion(val_outputs, val_Y).item()
        
    return final_train_loss, val_loss, train_time, inf_time_ms

def run_custom_targets_benchmark():
    """
    Main benchmark orchestrator. Evaluates models against highly complex, non-linear
    multiplicative and comparative targets.
    
    Ana benchmark yöneticisi. Modelleri son derece karmaşık, doğrusal olmayan 
    çarpımsal ve karşılaştırmalı hedeflere karşı test eder.
    """
    dim = 5       
    seq_len = 8
    target_types = ['multiplicative', 'comparative']
    
    raw_data = generate_raw_synthetic_data(total_points=10000, dim=dim)
    
    print("\n[ RSNet vs RNN/LSTM/GRU/Mamba ] - Custom Target Benchmarks (Seq_len: 8)...")
    print("\n[ RSNet vs RNN/LSTM/GRU/Mamba ] - Özel Hedef Testleri Başlıyor (Dizi Uzunluğu: 8)...")
    
    for target_type in target_types:
        if target_type == 'multiplicative':
            print("\n================ TARGET 1 (HEDEF 1): MULTIPLICATIVE (ÇARPIMSAL: x[t]*x[t-1]*... ) ================")
        else:
            print("\n================ TARGET 2 (HEDEF 2): COMPARATIVE (ARDISIK FARKLAR: f(x[t] - x[t-1]) ) ================")
            
        print(f"{'Model':<10} | {'Train Loss':<15} | {'Val Loss':<15} | {'Inf. Time (ms)':<15} | {'Train Time'}")
        print("-" * 85)
        
        train_X, train_Y, val_X, val_Y = prepare_synthetic_data_with_custom_targets(raw_data, target_type)
        
        models = {
            'RSNet': CryptoPredictorRSNet(seq_len=seq_len, dim=dim),
            'RNN': CryptoPredictorRNN(dim, hidden_dim=32, rnn_type='RNN'),
            'GRU': CryptoPredictorRNN(dim, hidden_dim=32, rnn_type='GRU'),
            'LSTM': CryptoPredictorRNN(dim, hidden_dim=32, rnn_type='LSTM'),
            'Mamba': CryptoPredictorMamba(dim=dim, hidden_dim=32)
        }
        
        for name, model in models.items():
            t_loss, v_loss, t_time, inf_time = train_and_eval(model, name, train_X, train_Y, val_X, val_Y, epochs=150)
            print(f"{name:<10} | {t_loss:<15.4f} | {v_loss:<15.4f} | {inf_time:<15.4f} | {t_time:.2f} s")

if __name__ == '__main__':
    run_custom_targets_benchmark()
