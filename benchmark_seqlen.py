import torch
import torch.nn as nn
import torch.optim as optim
import time
from rs_net import RSNet

class CryptoPredictorRSNet(nn.Module):
    """
    Wrapper class adapting the RSNet architecture for sequence prediction tasks.
    RSNet mimarisini dizi tahmini görevleri için uyarlayan sarmalayıcı sınıf.
    
    It maps the final RSNet node representation to a single scalar value.
    Nihai RSNet düğüm temsilini tek bir skaler değere dönüştürür (haritalandırır).
    """
    def __init__(self, seq_len, dim):
        super().__init__()
        self.rsnet = RSNet(num_inputs=seq_len, input_dim=dim)
        self.fc = nn.Linear(dim, 1) 
        
    def forward(self, x_list):
        rs_out = self.rsnet(x_list)
        return self.fc(rs_out)

def generate_raw_synthetic_data(total_points=10000, dim=5):
    """
    Generates a deterministic synthetic time-series dataset.
    Combines sinusoidal waves of varying frequencies with random walks and noise.
    
    Belirli bir desene sahip sentetik zaman serisi veri seti üretir.
    Değişken frekanslı sinüs dalgalarını, rastgele yürüyüş (random walk) ve gürültü ile birleştirir.
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
    and prepares corresponding target returns.
    
    Ham sentetik veriyi `seq_len` boyutunda örtüşen pencerelere böler
    ve karşılık gelen hedef çıktıları (getirileri) hazırlar.
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
    
    # Split into Train (80%) and Validation (20%)
    # Eğitim (%80) ve Doğrulama (%20) olarak böl
    split_idx = int(len(X) * 0.8)
    train_X, val_X = X[:split_idx], X[split_idx:]
    train_Y, val_Y = Y[:split_idx], Y[split_idx:]
    
    return train_X, train_Y, val_X, val_Y

def train_and_eval(model, train_X, train_Y, val_X, val_Y, epochs=100, lr=0.005):
    """
    Executes training and evaluation specifically for RSNet, recording metrics like
    training loss, validation loss, total training time, and inference time.
    
    Özellikle RSNet için eğitim ve değerlendirmeyi yürütür; eğitim kaybı (loss), 
    doğrulama kaybı, toplam eğitim süresi ve çıkarım (inference) süresi gibi metrikleri kaydeder.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # RSNet expects input as a list of Tensors / RSNet girdileri Tensör listesi olarak bekler
    inputs_train = [train_X[:, i, :] for i in range(train_X.size(1))]
    inputs_val = [val_X[:, i, :] for i in range(val_X.size(1))]
        
    # --- Inference Time Measurement / Çıkarım Süresi Ölçümü ---
    model.eval()
    with torch.no_grad():
        _ = model(inputs_val) # GPU/CPU Warmup / Isınma
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

def run_seqlen_benchmark():
    """
    Evaluates RSNet independently across exponentially increasing sequence lengths.
    RSNet'i üssel olarak artan dizi uzunlukları (sequence lengths) boyunca bağımsız olarak değerlendirir.
    """
    dim = 5       
    seq_lens = [4, 8, 16, 32, 64] 
    
    raw_data = generate_raw_synthetic_data(total_points=10000, dim=dim)
    
    print("\n[ Synthetic Dataset Generated ] - Commencing RSNet Sequence Length Benchmark...")
    print("\n[ Sentetik Veri Üretildi ] - RSNet Dizi Uzunluğu (Seq_len) Testi Başlıyor...")
    print(f"\n{'Seq_Len':<10} | {'Train Loss':<15} | {'Val Loss':<15} | {'Inf. Time (ms)':<15} | {'Train Time'}")
    print("-" * 80)
    
    for seq_len in seq_lens:
        train_X, train_Y, val_X, val_Y = prepare_synthetic_data_for_seq_len(raw_data, seq_len)
        model = CryptoPredictorRSNet(seq_len=seq_len, dim=dim)
        t_loss, v_loss, t_time, inf_time = train_and_eval(model, train_X, train_Y, val_X, val_Y, epochs=150)
        print(f"{seq_len:<10} | {t_loss:<15.4f} | {v_loss:<15.4f} | {inf_time:<15.4f} | {t_time:.2f} s")

if __name__ == '__main__':
    run_seqlen_benchmark()
