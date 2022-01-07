import torch
import pytorch_lightning as pl
from torch import Tensor, nn
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np

X,y = load_boston(return_X_y=True)

class SklearnDataset(Dataset):
    def __init__(self, X:np.ndarray, y:np.ndarray):
        super().__init__()
        scaler = MinMaxScaler()

        scaler.fit(X)
        self.X = scaler.transform(X)
        self.Y = y

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        return x, y

bostonds = SklearnDataset(X,y)
train_loader = DataLoader(bostonds, batch_size=32, shuffle=True, drop_last=True)

class LinRegModel(pl.LightningModule):
    def __int__(self, input_dim:int):
        super().__init__()
        self.linear = nn.Linear(in_features=13, out_features=1, bias=True)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

trainer = pl.Trainer()
model = LinRegModel(input_dim=13)
trainer.fit(model, train_loader)
