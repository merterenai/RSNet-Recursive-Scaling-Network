import torch
import torch.nn as nn
import torch.optim as optim
import time
from rs_net import RSNet
from mamba_minimal import CryptoPredictorMamba

class CryptoPredictorRSNet(nn.Module):
    """
    Wrapper class adapting the RSNet architecture for sequence prediction tasks.
    RSNet mimarisini dizi tahmini görevleri için uyarlayan sarmalayıcı sınıf.
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
    
    Serves as the baseline for evaluating RSNet's performance across different sequence lengths.
    Farklı dizi uzunluklarında RSNet'in performansını değerlendirmek için temel (baseline) görevi görür.
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
    Generates a deterministic synthetic time-series dataset.
    Combines sinusoidal waves of varying frequencies with random walks and noise.
    Using a single dataset ensures all models and sequence lengths are evaluated fairly.
    
    Belirli bir desene sahip sentetik zaman serisi veri seti üretir.
    Değişken frekanslı sinüs dalgalarını rastgele yürüyüş ve gürültü ile birleştirir.
    Tek bir veri seti kullanılması, tüm modellerin ve dizi uzunluklarının adil bir şekilde test edilmesini sağlar.
    """
    t = torch.linspace(0, 100, total_points)
    raw_data = torch.zeros(total_points, dim)
    for d in range(dim):
        signal = torch.sin(t * (d + 1) * 0.5) + torch.randn(total_points) * 0.1
        signal += torch.cumsum(torch.randn(total_points) * 0.05, dim=0)
        raw_data[:, d] = signal
    return raw_data

def prepare_synthetic_data_for_seq_len(raw_data, seq_len):
    """
    Slices the raw synthetic data into overlapping windows of size `seq_len`
    and prepares corresponding targets.
    
    Ham sentetik veriyi `seq_len` boyutunda örtüşen pencerelere böler
    ve karşılık gelen hedefleri hazırlar.
    """
    X_list = []
    Y_list = []
    
    for i in range(len(raw_data) - seq_len):
        window = raw_data[i : i + seq_len]
        target = raw_data[i + seq_len, 0] 
        
        # Per-window normalization / Pencere bazlı normalizasyon
        w_mean = window.mean(dim=0, keepdim=True)
        w_std = window.std(dim=0, keepdim=True) + 1e-8
        window_norm = (window - w_mean) / w_std
        target_norm = (target - w_mean[0, 0]) / w_std[0, 0]
        
        X_list.append(window_norm)
        Y_list.append(torch.tensor([target_norm], dtype=torch.float32))
        
    X = torch.stack(X_list)
    Y = torch.stack(Y_list)
    
    # Split into Train (80%) and Validation (20%) / Eğitim (%80) ve Doğrulama (%20)
    split_idx = int(len(X) * 0.8)
    train_X, val_X = X[:split_idx], X[split_idx:]
    train_Y, val_Y = Y[:split_idx], Y[split_idx:]
    
    return train_X, train_Y, val_X, val_Y

def train_and_eval(model, model_name, train_X, train_Y, val_X, val_Y, epochs=100, lr=0.005):
    """
    Executes training and evaluation for a given model, recording metrics like
    training loss, validation loss, total training time, and inference time per batch.
    
    Belirli bir model için eğitim ve değerlendirmeyi yürütür; eğitim kaybı, 
    doğrulama kaybı, toplam eğitim süresi ve grup (batch) başına çıkarım süresi gibi metrikleri kaydeder.
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
        _ = model(inputs_val) # GPU/CPU Warmup / Isınma
        
        start_inf = time.perf_counter()
        for _ in range(10): # Average over 10 runs / 10 çalışma üzerinden ortalama al
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
        
        # Gradient Clipping / Gradyan Kırpma
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

def run_seqlen_benchmark_all():
    """
    Main benchmark orchestrator. Evaluates RSNet, RNN, GRU, LSTM, and Mamba
    across exponentially increasing sequence lengths to demonstrate RSNet's O(log N) depth advantage.
    
    Ana benchmark yöneticisi. RSNet'in O(log N) derinlik avantajını göstermek için RSNet, RNN, GRU, LSTM ve Mamba'yı
    üssel olarak artan dizi uzunlukları (sequence lengths) boyunca değerlendirir.
    """
    dim = 5       
    seq_lens = [4, 8, 16, 32, 64, 128] 
    
    # Generate the base dataset once to ensure fairness
    # Adil bir karşılaştırma için taban veri setini yalnızca bir kez üret
    raw_data = generate_raw_synthetic_data(total_points=10000, dim=dim)
    
    print("\n[ Synthetic Dataset Generated ] - Commencing Sequence Length Scaling Benchmark...")
    print("[ Sentetik Veri Üretildi ] - Dizi Uzunluğu (Seq_len) Ölçekleme Testi Başlıyor...")
    
    for seq_len in seq_lens:
        print(f"\n================ SEQUENCE LENGTH (DİZİ UZUNLUĞU): {seq_len} ================")
        print(f"{'Model':<10} | {'Train Loss':<15} | {'Val Loss':<15} | {'Inf. Time (ms)':<15} | {'Train Time'}")
        print("-" * 80)
        
        train_X, train_Y, val_X, val_Y = prepare_synthetic_data_for_seq_len(raw_data, seq_len)
        
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
    run_seqlen_benchmark_all()
