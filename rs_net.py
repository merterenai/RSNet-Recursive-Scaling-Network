import torch
import torch.nn as nn
import math

class RSBlock(nn.Module):
    """
    RSBlock: The fundamental computing unit of the RSNet architecture.
    RSBlock: RSNet mimarisinin temel hesaplama birimidir.
    
    This block fuses two input representations (X1, X2) using a custom 
    non-linear multiplicative-ratio formula, controlled by learnable weight matrices.
    Bu blok, iki girdiyi (X1, X2) öğrenilebilir ağırlık matrisleriyle kontrol edilen, 
    özgün ve doğrusal olmayan bir çarpımsal-oran formülü kullanarak birleştirir.
    """
    def __init__(self, input_dim):
        super(RSBlock, self).__init__()
        # Learnable weight matrices for the input projections and output transformation
        # Girdi izdüşümleri ve çıktı dönüşümü için öğrenilebilir ağırlık matrisleri
        self.W1 = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        self.W2 = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        self.W3 = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        
        # Epsilon constant to ensure numerical stability during division
        # Bölme işleminde matematiksel kararlılığı (sıfıra bölünmeyi) sağlamak için Epsilon sabiti
        self.eps = 1e-7

    def forward(self, x1, x2):
        """
        Forward pass applying the custom RSNet formula.
        Özgün RSNet formülünü uygulayan ileri besleme (forward pass) adımı.
        
        Mathematical Formulation / Matematiksel Formülasyon:
        1. Linear Projections (Doğrusal İzdüşüm):  Y1 = X1 @ W1,  Y2 = X2 @ W2 (Matrix Multiplication / Matris Çarpımı)
        2. Multiplicative Fusion (Çarpımsal Birleşim): F1 = Y1 ⊙ Y2 (Element-wise / Eleman Bazlı)
        3. Dynamic Scaling (Dinamik Oranlama): F2 = F1 / (|Y1 + Y2| + eps)
        4. Activation (Aktivasyon): X3 = LeakyReLU(F2, negative_slope=0.01)
        5. Output Projection (Çıktı İzdüşümü): Y3 = X3 @ W3
        6. Final Activation (Son Aktivasyon): out = LeakyReLU(Y3, negative_slope=0.01)
        """
        # Step 1: Linear projections for both inputs
        # Adım 1: Her iki girdi için doğrusal izdüşümler
        y1 = torch.matmul(x1, self.W1)
        y2 = torch.matmul(x2, self.W2)

        # Step 2: Multiplicative feature fusion
        # Adım 2: Çarpımsal özellik birleşimi
        f1 = y1 * y2

        # Step 3: Dynamic ratio scaling (Prevents exploding values and normalizes magnitude)
        # Adım 3: Dinamik oranlama (Değerlerin patlamasını önler ve büyüklükleri dengeler)
        denominator = torch.abs(y1 + y2) + self.eps
        f2 = f1 / denominator

        # Step 4: First non-linear activation (LeakyReLU-like thresholding)
        # Adım 4: İlk doğrusal olmayan aktivasyon (LeakyReLU benzeri eşikleme)
        x3 = torch.where(f2 > 0, f2, 0.01 * f2)

        # Step 5: Final feature transformation
        # Adım 5: Nihai özellik dönüşümü
        y3 = torch.matmul(x3, self.W3)

        # Step 6: Final non-linear activation
        # Adım 6: Nihai doğrusal olmayan aktivasyon
        out = torch.where(y3 > 0, y3, 0.01 * y3)
        
        return out


class RSNet(nn.Module):
    """
    RSNet: Recursive Scaling Network
    RSNet: Özyinelemeli Ölçekleme Ağı (Recursive Scaling Network)
    
    A hierarchical, tree-based neural network architecture designed for sequence modeling. 
    Instead of processing sequences sequentially (O(N) sequential bottleneck like RNNs), 
    RSNet reduces the sequence logarithmically by pairing adjacent elements at each depth level.
    
    Dizi modelleme için tasarlanmış hiyerarşik, ağaç tabanlı bir sinir ağı mimarisi.
    Dizileri ardışık olarak işlemek yerine (RNN'lerdeki O(N) darboğazı gibi),
    RSNet her derinlik seviyesinde yan yana olan elemanları eşleştirerek diziyi logaritmik olarak küçültür.
    
    Time Complexity (Zaman Karmaşıklığı): O(log2(N)) depth for a sequence of length N.
    """
    def __init__(self, num_inputs, input_dim):
        super(RSNet, self).__init__()
        
        # Validate that the sequence length allows for a perfect binary tree
        # Dizi uzunluğunun kusursuz bir ikili ağaç (binary tree) oluşturmaya uygun olup olmadığını doğrula
        if num_inputs < 2 or (num_inputs & (num_inputs - 1)) != 0:
            raise ValueError(f"RSNet requires 'num_inputs' to be a power of 2 (e.g., 2, 4, 8, 16...). Got: {num_inputs} | RSNet için 'num_inputs' değeri 2'nin üssü olmalıdır.")
            
        # Determine the number of hierarchical levels (depth of the tree)
        # Hiyerarşik seviyelerin (ağacın derinliğinin) sayısını belirle
        self.depth = int(math.log2(num_inputs))
        
        # Instantiate a distinct RSBlock for each depth level.
        # This layer-wise weight sharing allows the network to learn different abstraction 
        # logic at different stages of the hierarchy (e.g., local patterns vs. global trends).
        
        # Her derinlik seviyesi için ayrı bir RSBlock oluştur.
        # Bu katman bazlı ağırlık paylaşımı, ağın hiyerarşinin farklı aşamalarında farklı 
        # soyutlama mantıkları öğrenmesine olanak tanır (Örn: yerel desenler vs genel eğilimler).
        self.layers = nn.ModuleList([
            RSBlock(input_dim) for _ in range(self.depth)
        ])

    def forward(self, inputs):
        """
        inputs: A list of tensors [X1, X2, ..., Xn] where n == num_inputs.
        inputs: Boyutu num_inputs olan [X1, X2, ..., Xn] şeklinde bir tensör listesi.
        
        Returns: A single output tensor representing the aggregated root of the tree.
        Döndürür: Ağacın birleştirilmiş kökünü temsil eden tek bir çıktı tensörü.
        """
        curr_layer_data = inputs
        
        # Hierarchically reduce the sequence level by level
        # Diziyi seviye seviye hiyerarşik olarak küçült
        for layer_block in self.layers:
            next_layer_data = []
            
            # Pair adjacent elements and process them through the current level's RSBlock
            # Yan yana olan elemanları eşleştir ve o anki seviyenin RSBlock'u üzerinden geçir
            for i in range(0, len(curr_layer_data), 2):
                res = layer_block(curr_layer_data[i], curr_layer_data[i+1])
                next_layer_data.append(res)
                
            # Move up the tree to the next depth level
            # Ağaçta bir sonraki derinlik seviyesine çık
            curr_layer_data = next_layer_data
            
        # Return the final root node output
        # Nihai kök düğümünün çıktısını döndür
        return curr_layer_data[0]