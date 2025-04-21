import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


# 定义一个简单的跨注意力机制
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert d_model % self.n_heads == 0

        self.depth = d_model // self.n_heads
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        Q = self.WQ(queries)  # (batch_size, seq_len, d_model)
        K = self.WK(keys)  # (batch_size, seq_len, d_model)
        V = self.WV(values)  # (batch_size, seq_len, d_model)

        Q = self.split_heads(Q)  # (batch_size, n_heads, seq_len, depth)
        K = self.split_heads(K)  # (batch_size, n_heads, seq_len, depth)
        V = self.split_heads(V)  # (batch_size, n_heads, seq_len, depth)

        # Scaled dot-product attention
        Z = torch.matmul(Q, K.transpose(-2, -1)) / (self.depth ** 0.5)
        attention_weights = nn.functional.softmax(Z, dim=-1)
        output = torch.matmul(attention_weights, V)

        output = self.combine_heads(output)  # (batch_size, seq_len, d_model)
        return output

    def split_heads(self, x):
        """Split the last dimension into (n_heads, depth)."""
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)

    def combine_heads(self, x):
        """Combine the last two dimensions back into (d_model)."""
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)


# 定义整个模块
class PVAEModel(nn.Module):
    def __init__(self, num_classes, d_model, n_heads, J, M):
        super(PVAEModel, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.cross_attention = CrossAttention(d_model, n_heads)
        self.fc = nn.Linear(d_model, num_classes)
        self.J = J
        self.M = M

    def forward(self, x, S):
        # 提取视觉特征
        x = self.resnet(x)
        # 跨注意力机制
        x = self.cross_attention(x, S, S)
        # 最终分类
        x = self.fc(x)
        return x


# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PVAEModel(num_classes=1000, d_model=2048, n_heads=1, J=4, M=4).to(device)

# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练循环
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs, S)  # S 应该是预先定义好的语义属性嵌入
        loss = criterion(outputs, labels)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# 这里需要定义数据加载器和损失函数，并且 S 需要提前准备好
criterion = nn.CrossEntropyLoss()
train_dataloader = ...  # 你需要定义训练数据加载器
valid_dataloader = ...  # 你需要定义验证数据加载器

# 开始训练
num_epochs = 50
for epoch in range(num_epochs):
    train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')