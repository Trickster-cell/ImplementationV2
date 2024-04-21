from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from DataLoader import data_transform, transform, encode_segmap, n_classes
import torch

class SegFormerDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = data_transform(root='../cityscapes/', split='train', mode='fine', target_type='semantic', transforms=transform)        
        self.val_dataset = data_transform(root='../cityscapes/', split='val', mode='fine', target_type='semantic', transforms=transform)
        # print(self.train_dataset.shape)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)



# %%

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingLR
from SegFormer import SegFormer

class SegFormerModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SegFormer(
            in_channels=3, #ok
            widths=[64, 128, 320, 512], # changed the third depth C3 to 320 from 256(as in B5)
            depths=[3, 6, 40, 3], #change according to the B5
            all_num_heads=[1, 2, 5, 8], #changed the third num head to 5 from 3 (stage 3 change)
            patch_sizes=[7, 3, 3, 3], #ok
            overlap_sizes=[4, 2, 2, 2], #ok
            reduction_ratios=[8, 4, 2, 1], #ok
            mlp_expansions=[4, 4, 4, 4], #ok
            decoder_channels=768,
            scale_factors=[8, 4, 2, 1],
            num_classes=20,
        )
        self.criterion = smp.losses.FocalLoss(mode='multiclass')
        # self.criterion = smp.losses.LovaszLoss(mode='multiclass')

        self.metrics = torchmetrics.JaccardIndex(num_classes=n_classes, task='multiclass')

    def forward(self,x):
        return self.model(x)

    def process(self, image, segment):
        out=self(image)
        # print(out.shape, segment.shape)
        segment = encode_segmap(segment)
        # print(segment.shape)
        loss= self.criterion(out, segment.long())
        iou = self.metrics(out, segment)
        return loss, iou

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00006)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor' : 'val_loss'}

    def training_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss


model = SegFormerModel()
datamodule = SegFormerDataModule(batch_size=2)

checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='file', save_last = True)
# Define the early stopping callback

from pytorch_lightning.callbacks import EarlyStopping
early_stop_callback = EarlyStopping(monitor='val_loss', patience=15, verbose=True, mode='min')


# %%
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger("logs/CosineAnnV0/", name = "SegFormer_v2_epoch501_CosineAnnV0zLoss")


trainer = Trainer(max_epochs=100,
                  accelerator="cuda" if torch.cuda.is_available() else "cpu",
                  callbacks=[checkpoint_callback, early_stop_callback],
                  num_sanity_val_steps=0,
                  logger = tb_logger
                  )

# %%
trainer.fit(model, datamodule=datamodule)
# Loading the best model from checkpoint
best_model = SegFormerModel.load_from_checkpoint(checkpoint_callback.best_model_path)

# Assuming you have trained your model and it's stored in the variable `best_model`

# Define the file path where you want to save the model weights
weights_path = "segformer_100epochs_model_weightsCosineAnnV0Loss.pth"

# Save the model weights
# torch.save(best_model.state_dict(), weights_path)

# Optionally, you can also save the entire model
torch.save(best_model, 'entire_model.pth')
