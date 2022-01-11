import torch
import torchvision.transforms as T
from trainmodule import Network

class DeployPractice:
    def __init__(self):
        self.model = Network.load_from_checkpoint('./checkpoint/practice.ckpt')
        self.model.to('cuda:1')
        self.model.eval()
        self.transform = T.Compose([
            T.Resize([224,224]),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
    def pre_processing(self, image):
        return self.transform(image)
    
    def classification(self, x):
        x = x[None].to('cuda:1')
        logit = self.model(x)
        
        # post processing
        y_hat = torch.argmax(logit).item()
        return y_hat
    

        