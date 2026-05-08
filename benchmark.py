import torch
import torch.nn as nn
import torch.optim as optim
import time
from rs_net import RSNet

class BaselineRNN(nn.Module):
    """
    Standard Recurrent Neural Networks (RNN, LSTM, GRU) wrapper for baseline comparison.
    Karşılaştırma (baseline) için Standart Tekrarlayan Sinir Ağları (RNN, LSTM, GRU) sarmalayıcısı.
    
    Extracts the final hidden state to predict a sequence-level target.
    Dizi seviyesinde bir hedefi tahmin etmek için son gizli durumu (hidden state) çıkarır.
    """
    def __init__(self, dim, hidden_dim, rnn_type='LSTM'):
        super(BaselineRNN, self).__init__()
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(dim, hidden_dim, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(dim, hidden_dim, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(dim, hidden_dim, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        # x boyutu: (grup_boyutu, dizi_uzunlugu, boyut)
        out, _ = self.rnn(x)
        
        # We only care about the output of the last time step
        # Sadece son zaman adımının çıktısıyla ilgileniyoruz
        last_out = out[:, -1, :]
        return self.fc(last_out)

def generate_synthetic_data(num_samples=1000, seq_len=8, dim=64):
    """
    Generates synthetic sequential data.
    Sentetik ardışık (dizisel) veri üretir.
    
    The task is to predict the sum of the sequence elements (Linear Sum Task).
    Görev, dizideki elemanların toplamını tahmin etmektir (Doğrusal Toplam Görevi).
    """
    X = torch.randn(num_samples, seq_len, dim)
    
    # Ground Truth: Sum along the sequence length
    # Gerçek Değer: Dizi uzunluğu boyunca toplam
    Y = torch.sum(X, dim=1) 
    return X, Y

def train_and_evaluate(model, model_name, X_train, Y_train, X_val, Y_val, epochs=50, lr=0.01):
    """
    Training and evaluation protocol.
    Eğitim ve değerlendirme protokolü.
    
    Records Mean Squared Error (MSE) and inference times for cross-model comparison.
    Modeller arası karşılaştırma için Ortalama Kare Hata (MSE) ve çıkarım sürelerini kaydeder.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Format inputs based on architecture type
    # Mimari türüne göre girdileri biçimlendir
    if model_name == 'RSNet':
        inputs_train = [X_train[:, i, :] for i in range(X_train.size(1))]
        inputs_val = [X_val[:, i, :] for i in range(X_val.size(1))]
    else:
        inputs_train = X_train
        inputs_val = X_val
        
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        
        # Apply gradient clipping to stabilize training
        # Eğitimi stabilize etmek için gradyan kırpma (clipping) uygula
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    train_time = time.time() - start_time
    final_train_loss = loss.item()
    
    # Inference Time Measurement
    # Çıkarım Süresi (Inference Time) Ölçümü
    model.eval()
    with torch.no_grad():
        _ = model(inputs_val) # Warmup / Isınma Turu
        
        start_inf = time.perf_counter()
        for _ in range(10):
            val_outputs = model(inputs_val)
        inf_time_ms = ((time.perf_counter() - start_inf) / 10.0) * 1000
        
        val_loss = criterion(val_outputs, Y_val).item()
        
    return final_train_loss, val_loss, train_time, inf_time_ms

def run_benchmark():
    """
    Orchestrates the general benchmark comparing RSNet against RNN, GRU, and LSTM
    on a synthetic sequence aggregation task.
    RSNet'i sentetik dizi toplama görevi üzerinde RNN, GRU ve LSTM'ye karşı 
    kıyaslayan genel benchmark testini yönetir.
    """
    num_samples = 2000
    seq_len = 8    
    dim = 32       
    epochs = 100
    
    # Data Preparation / Veri Hazırlığı
    X, Y = generate_synthetic_data(num_samples, seq_len, dim)
    
    # Split into Train (80%) and Validation (20%)
    # Veriyi Eğitim (%80) ve Doğrulama (%20) olarak böl
    split = int(num_samples * 0.8)
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    
    models = {
        'RSNet': RSNet(num_inputs=seq_len, input_dim=dim),
        'RNN': BaselineRNN(dim, hidden_dim=64, rnn_type='RNN'),
        'GRU': BaselineRNN(dim, hidden_dim=64, rnn_type='GRU'),
        'LSTM': BaselineRNN(dim, hidden_dim=64, rnn_type='LSTM')
    }
    
    print(f"--- RSNet vs. Traditional Recurrent Models Benchmark ---")
    print(f"--- RSNet ve Geleneksel Tekrarlayan Modeller Karşılaştırması ---")
    print(f"Dataset: Synthetic (Sum), Samples: {num_samples}, Seq_Len: {seq_len}, Dim: {dim}")
    print(f"\n{'Model':<10} | {'Train Loss (MSE)':<18} | {'Val Loss (MSE)':<18} | {'Inference (ms)':<15} | {'Train Time'}")
    print("-" * 85)
    
    for name, model in models.items():
        t_loss, v_loss, t_time, inf_time = train_and_evaluate(
            model, name, X_train, Y_train, X_val, Y_val, epochs=epochs
        )
        print(f"{name:<10} | {t_loss:<18.4f} | {v_loss:<18.4f} | {inf_time:<15.4f} | {t_time:.2f} s")

if __name__ == '__main__':
    run_benchmark()
