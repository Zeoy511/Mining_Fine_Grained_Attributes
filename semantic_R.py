import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from ot import sinkhorn


class SemanticAttributeReconstructionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(SemanticAttributeReconstructionModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 768  # 视觉特征的维度
model = SemanticAttributeReconstructionModule(input_dim).to(device)

# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=0.05)


# 训练循环
def train_model(model, dataloader, criterion, optimizer, device, gamma1=0.01, gamma2=0.01, alpha=0.5, beta=0.5):
    model.train()
    running_loss = 0.0
    for i, (visual_features, semantic_features) in enumerate(dataloader):
        visual_features = visual_features.to(device)
        semantic_features = semantic_features.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        recon_visual_features, latent_features = model(visual_features)
        recon_semantic_features = model.decoder(model.encoder(semantic_features))

        # 计算损失
        recon_loss = criterion(recon_visual_features, visual_features) + criterion(recon_semantic_features,
                                                                                   semantic_features)

        # 构建混淆矩阵
        Mvis = torch.mm(visual_features.T, recon_visual_features)
        Msem = torch.mm(semantic_features.T, recon_semantic_features)

        # 构建成本矩阵
        Cvis = 1 - Mvis
        Csem = 1 - Msem

        # 应用Sinkhorn算法得到传输计划
        mu = torch.ones(M) / M
        nu = torch.ones(M) / M
        Tvis = sinkhorn(mu, nu, Cvis.cpu().detach().numpy(), reg=gamma1, numItermax=1000)
        Tsem = sinkhorn(torch.ones(J) / J, torch.ones(J) / J, Csem.cpu().detach().numpy(), reg=gamma2, numItermax=1000)

        # 计算OT距离
        OT_V = torch.sum(torch.tensor(Tvis) * torch.tensor(Cvis))
        OT_S = torch.sum(torch.tensor(Tsem) * torch.tensor(Csem))

        # 结构对齐损失
        struct_loss = alpha * (OT_V + OT_S)

        # 总损失
        total_loss = recon_loss + beta * struct_loss

        # 反向传播与优化
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * visual_features.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# 这里需要定义数据加载器和损失函数
criterion = nn.MSELoss()
train_dataloader = ...  # 你需要定义训练数据加载器
valid_dataloader = ...  # 你需要定义验证数据加载器

# 开始训练
num_epochs = 50
for epoch in range(num_epochs):
    train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')