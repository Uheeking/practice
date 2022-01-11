# datamodule
# 이미지
from pytorch_lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as T
class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder('/data/datasets/k_fashion_detections/train', transform=transform)
        
        train_length = int(len(dataset)*.8)
        val_length = len(dataset) - train_length
        
        self.train_dataset, self.val_dataset = random_split(dataset=dataset, lengths=[train_length, val_length])
        self.batch_size = 128
        self.num_workers =4
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers,
                         pin_memory=True,
                         drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers,
                         pin_memory=True,
                         drop_last=True)