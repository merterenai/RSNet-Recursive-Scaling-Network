import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaMinimal(nn.Module):
    """
    MambaMinimal: A pure PyTorch implementation of the Mamba (State Space Model) architecture.
    MambaMinimal: Mamba (State Space Model) mimarisinin saf (pure) PyTorch implementasyonu.
    
    This minimal implementation avoids the complex custom CUDA kernel dependencies 
    (like causal-conv1d and mamba-ssm) that are notoriously difficult to compile on Windows.
    It provides a mathematically equivalent forward pass using PyTorch primitives, making it 
    perfect for benchmarking and cross-platform compatibility.
    
    Bu minimal uygulama, özellikle Windows'ta derlenmesi zor olan karmaşık CUDA çekirdek
    bağımlılıklarını (causal-conv1d ve mamba-ssm gibi) ortadan kaldırır. PyTorch temellerini
    kullanarak matematiksel olarak eşdeğer bir ileri besleme (forward pass) sunar ve böylece
    çapraz platform (cross-platform) uyumluluğu ile benchmark testleri için mükemmel hale gelir.
    """
    def __init__(self, d_model, d_state=16, expand=2, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(int(d_model / 16), 1)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=4,
            groups=self.d_inner,
            padding=3
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # S4D real initialization for the state transition matrix
        # Durum geçiş matrisi (state transition matrix) için S4D gerçek başlatması
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x shape: (Batch, Sequence Length, Feature Dimension)
        # x boyutu: (Grup Boyutu, Dizi Uzunluğu, Özellik Boyutu)
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_state, z = xz.chunk(2, dim=-1)
        
        # 1D Convolution over the sequence length
        # Dizi uzunluğu boyunca 1 Boyutlu Konvolüsyon (Evrişim)
        x_conv = x_state.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_state = F.silu(x_conv)
        
        # State Space parameters projections
        # Durum Uzayı (State Space) parametre izdüşümleri
        x_dbl = self.x_proj(x_state)
        dt, B_matrix, C_matrix = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        
        A = -torch.exp(self.A_log)
        
        y = torch.zeros(B, L, self.d_inner, device=x.device)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        
        # Selective Scan (Sequential loop for pure PyTorch compatibility)
        # Seçici Tarama (Saf PyTorch uyumluluğu için ardışık döngü)
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1)
            dA = torch.exp(dt_t * A)
            dB = dt_t * B_matrix[:, t].unsqueeze(1)
            
            h = dA * h + dB * x_state[:, t].unsqueeze(-1)
            y[:, t] = (h * C_matrix[:, t].unsqueeze(1)).sum(dim=-1) + self.D * x_state[:, t]
            
        y = y * F.silu(z)
        return self.out_proj(y)

class CryptoPredictorMamba(nn.Module):
    """
    Wrapper class to adapt the MambaMinimal model for sequence-to-scalar prediction tasks.
    MambaMinimal modelini dizi-skaler tahmini görevleri için uyarlayan sarmalayıcı sınıf.
    """
    def __init__(self, dim, hidden_dim=32):
        super().__init__()
        self.proj_in = nn.Linear(dim, hidden_dim)
        self.mamba = MambaMinimal(d_model=hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Maps raw features to the Mamba dimension and extracts the final state for prediction
        # Ham özellikleri Mamba boyutuna dönüştürür ve tahmin için son durumu (state) çıkarır
        x = self.proj_in(x)
        out = self.mamba(x)
        return self.fc(out[:, -1, :])
