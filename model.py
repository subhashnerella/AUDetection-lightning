
from lightning.pytorch.core import LightningModule
from transformers import  SwinForImageClassification
from torch import nn
import torch
from einops import rearrange

class SwinModel(LightningModule):
    def __init__(self,aus,
                 variant='base',
                 lr=1e-4,
                 ckpt_path=None,
                 loss_reduction='mean'):                 
        super().__init__()
        variants = {'base': 'microsoft/swin-base-patch4-window7-224',
                    'tiny': 'microsoft/Swin-tiny-patch4-window7-224'}
        self.model = SwinForImageClassification.from_pretrained(variants[variant])
        self.AUs = aus
        num_aus = len(aus)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_aus)
        self.criterion = nn.BCEWithLogitsLoss(reduction = loss_reduction)
        self.learning_rate = lr
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self,path):
        sd = torch.load(path,map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
        self.load_state_dict(sd,strict=False)

    def forward(self, x):
        outputs= self.model(x)
        return outputs.logits
    
    def compute_loss(self,outputs,labels):
        loss = self.criterion(outputs, labels)
        mask = labels != -1
        loss = loss * mask
        loss = loss.sum() / mask.sum()
        return loss
    
    def common_step(self, batch, batch_idx):
        x = batch['image']
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        outputs = self(x)
        labels = batch['aus']
        loss = self.compute_loss(outputs,labels)
        return loss,outputs

    def training_step(self, batch, batch_idx):
        loss,outputs = self.common_step(batch, batch_idx)
        self.log('train/loss', loss)
        return {'loss':loss,'logits':outputs}
    
    def validation_step(self, batch, batch_idx):
        loss,outputs = self.common_step(batch, batch_idx)
        self.log('val/loss', loss)
        return {'loss':loss,'logits':outputs}
    
    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer
