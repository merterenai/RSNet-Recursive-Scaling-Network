import torch
import torch.nn as nn
import torch.optim as optim
import time
from rs_net import RSNet

def generate_dataset(dataset_type, num_samples, seq_len, dim):
    """
    Generates diverse synthetic datasets to evaluate the model's capacity to capture
    various non-linear and temporal patterns.
    Modelin çeşitli doğrusal olmayan ve zamansal desenleri yakalama kapasitesini 
    değerlendirmek için çok çeşitli sentetik veri setleri üretir.
    """
    X = torch.randn(num_samples, seq_len, dim)
    
    if dataset_type == 1:
        # 1. Linear Sum - Basic sequence aggregation
        # 1. Doğrusal Toplam - Temel dizi birleştirme işlemi
        Y = torch.sum(X, dim=1)
    elif dataset_type == 2:
        # 2. Non-linear Sinusoidal mapping
        # 2. Doğrusal Olmayan Sinüs Dönüşümü
        Y = torch.sum(X, dim=1) + torch.sin(X[:, 0, :])
    elif dataset_type == 3:
        # 3. Max Pooling - Capturing extreme values
        # 3. Maksimum Ortaklama - Aşırı (uç) değerleri yakalama
        Y, _ = torch.max(X, dim=1)
    elif dataset_type == 4:
        # 4. Mean Pooling - Smoothing aggregation
        # 4. Ortalama Ortaklama - Pürüzsüzleştirici birleştirme
        Y = torch.mean(X, dim=1)
    elif dataset_type == 5:
        # 5. Temporal Differences - Modeling volatility
        # 5. Zamansal Farklar - Volatiliteyi (oynaklığı) modelleme
        Y = torch.sum(X[:, 1:, :] - X[:, :-1, :], dim=1)
    elif dataset_type == 6:
        # 6. Sum of Squares
        # 6. Karelerin Toplamı
        Y = torch.sum(X**2, dim=1)
    elif dataset_type == 7:
        # 7. Scaled Product - Tests the network's ability to handle multiplicative logic
        # 7. Ölçeklendirilmiş Çarpım - Ağın çarpımsal mantığı işleme yeteneğini test eder
        Y = torch.prod(X / 2.0, dim=1)
    elif dataset_type == 8:
        # 8. Cosine and Absolute Value Complexity
        # 8. Kosinüs ve Mutlak Değer Karmaşası
        Y = torch.sum(torch.cos(X) + torch.abs(X), dim=1)
    elif dataset_type == 9:
        # 9. Minimum Pooling - Minimum (Min)
        Y, _ = torch.min(X, dim=1)
    elif dataset_type == 10:
        # 10. First and Last Element - İlk ve Son Eleman
        Y = X[:, 0, :] + X[:, -1, :]
    else:
        Y = torch.sum(X, dim=1)
        
    return X, Y

def train_and_evaluate(model, X_train, Y_train, X_val, Y_val, epochs=50, lr=0.01):
    """
    Executes the training and validation loops specifically formatted for RSNet.
    RSNet için özel olarak biçimlendirilmiş eğitim ve doğrulama döngülerini çalıştırır.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # RSNet expects input as a list of Tensors rather than a single sequence Tensor
    # RSNet girdileri tek bir dizi tensörü yerine, Tensörlerin bir listesi olarak bekler
    inputs_train = [X_train[:, i, :] for i in range(X_train.size(1))]
    inputs_val = [X_val[:, i, :] for i in range(X_val.size(1))]
    
    start_time = time.time()
    
    # Training Loop / Eğitim Döngüsü
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        
        # Gradient clipping prevents the exploding gradient problem in deep recursive structures
        # Gradyan kırpma (clipping), derin özyinelemeli yapılardaki gradyan patlaması sorununu önler
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    train_time = time.time() - start_time
    final_train_loss = loss.item()
    
    # Validation Phase / Doğrulama Aşaması
    model.eval()
    with torch.no_grad():
        val_outputs = model(inputs_val)
        val_loss = criterion(val_outputs, Y_val).item()
        
    return final_train_loss, val_loss, train_time

def run_comprehensive_benchmark():
    """
    Orchestrates the 10-task comprehensive benchmark for the RSNet architecture.
    RSNet mimarisi için 10 görevlik kapsamlı benchmark testini yönetir.
    """
    num_samples = 1000
    seq_len = 8    
    dim = 16       
    epochs = 100
    
    print(f"--- RSNet Comprehensive Benchmark ---")
    print(f"--- RSNet Kapsamlı Benchmark Testi ---")
    print(f"Sequence Length: {seq_len}, Feature Dim: {dim}, Epochs: {epochs}")
    print(f"{'Task Name (Görev Adı)':<22} | {'Train Loss':<15} | {'Val Loss':<15} | {'Train Time'}")
    print("-" * 72)
    
    task_names = [
        "Doğrusal Toplam", "Sinüs Kombinasyonu", "Maksimum", "Ortalama", 
        "Ardışık Farklar", "Karesel Toplam", "Çarpımsal", "Cos + Mutlak Değer", 
        "Minimum (Min)", "İlk ve Son Eleman"
    ]
    
    for task_id in range(1, 11):
        # Generate task-specific dataset
        # Göreve özel veri setini oluştur
        X, Y = generate_dataset(task_id, num_samples, seq_len, dim)
        
        # Split into Train (80%) and Validation (20%)
        # Veriyi Eğitim (%80) ve Doğrulama (%20) olarak böl
        split = int(num_samples * 0.8)
        X_train, Y_train = X[:split], Y[:split]
        X_val, Y_val = X[split:], Y[split:]
        
        # Instantiate a fresh model for each task
        # Her görev için yepyeni (sıfırdan) bir model başlat
        model = RSNet(num_inputs=seq_len, input_dim=dim)
        
        t_loss, v_loss, t_time = train_and_evaluate(
            model, X_train, Y_train, X_val, Y_val, epochs=epochs
        )
        
        task_name = task_names[task_id - 1]
        print(f"{task_name:<22} | {t_loss:<15.4f} | {v_loss:<15.4f} | {t_time:.2f} s")

if __name__ == '__main__':
    run_comprehensive_benchmark()
