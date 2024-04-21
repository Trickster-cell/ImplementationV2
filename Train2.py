import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from snntorch import functional as SF

# import config
import torch
from SegFormer import SegFormer
from snntorch import utils
from DataLoader_copy import GetData

import gc

num_steps = 5
batch_size = 2

gc.collect()

torch.cuda.empty_cache()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")  

if torch.cuda.is_available():
    print("gpu being used for computing")
else:
    print("issues with gpu using cpu instead")

class accuracy_calc:
    '''
    calculate accuracy for the batch
    '''
    def __call__(self, spk_train, target_values):
        return SF.accuracy_rate(spk_train, target_values)

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
        self.criterion = SF.ce_rate_loss()
        self.metrics = accuracy_calc()

    def custom_forward_pass(self, data, num_steps= num_steps):
        '''
        i think this will take care of the num steps
        '''
        spk_rec = []

        utils.reset(self.model)  # resets hidden states for all LIF neurons in net
        data = data.permute(1,0,2,3,4)
        for step in range(num_steps):
            spk_out = self.model(data[step])
            spk_rec.append(spk_out)
        return torch.stack(spk_rec)


    # def custom_forward_pass(self, data, num_steps=config.num_steps): #3
    #     '''
    #     Forward pass with memory optimization.
    #     '''
    #     utils.reset(self.model)  # Reset hidden states for all LIF neurons in the network
    #     data = data.permute(1, 0, 2, 3, 4)
    #     spk_rec = torch.empty(num_steps, *self.model(data[0]).shape, device=data.device)
    #     for step in range(num_steps):
    #         spk_rec[step] = self.model(data[step])
    #     return spk_rec


    # def custom_forward_pass(self, data, num_steps=config.num_steps, batch_size=config.batch_size): #2
    #     spk_rec = []

    #     # Reset hidden states for all LIF neurons in the network
    #     utils.reset(self.model)

    #     # Permute data for batch processing
    #     data = data.permute(1, 0, 2, 3, 4)

    #     # with torch.no_grad():  # Disable gradient computation
    #     for start in range(0, num_steps, batch_size):
    #         end = min(start + batch_size, num_steps)
    #         batch_data = data[start:end]
    #         batch_spk_rec = []

    #         for step in range(batch_data.size(0)):
    #             spk_out = self.model(batch_data[step])
    #             batch_spk_rec.append(spk_out.detach())

    #         spk_rec.extend(batch_spk_rec)

    #     return torch.stack(spk_rec)


    def forward(self,x):
        return self.custom_forward_pass(x)

    def process(self, image, targets):
        # print(f"******{image.shape}")
        # torch.cuda.empty_cache()
        spk_rec = self(image)
        # spk_rec = spk_rec.detach()  # Create a detached copy to avoid inplace operations
        spk_rec = spk_rec.reshape(spk_rec.shape[0], -1, spk_rec.shape[2])
        # spk_rec.requires_grad = True # reshaped tensor used for gradient computation
        targets_reshape = targets.reshape(-1)

        loss_val = self.criterion(spk_rec, targets_reshape)
        acc = self.metrics(spk_rec, targets_reshape)
        # torch.cuda.empty_cache()
        return loss_val, acc

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience, verbose=True)
        return {'optimizer': optimizer, #'lr_scheduler': scheduler,
                'monitor' : 'val_loss'}




    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        train_image, train_segment = batch
        train_loss, train_accuracy = self.process(train_image, train_segment)
        # torch.cuda.empty_cache()
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss



    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        val_image, val_segment = batch
        val_loss, val_accuracy = self.process(val_image, val_segment)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # test_acc = self.batch_accuracy(test_loader, net, num_steps)
        return val_loss


model = SegFormerModel()
datamodule = GetData(batch_size=batch_size)

checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='file', save_last = True)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger("logs/", name = "Spiking_SegFormer_epoch")


trainer = Trainer(max_epochs=10,
                  accelerator="cuda" if torch.cuda.is_available() else "cpu",
                  callbacks=[checkpoint_callback],
                  num_sanity_val_steps=0,
                  logger = tb_logger,
                  )

trainer.fit(model, datamodule=datamodule)
# Loading the best model from checkpoint
best_model = SegFormerModel.load_from_checkpoint(checkpoint_callback.best_model_path)

# Define the file path where you want to save the model weights
weights_path = "hopes.pth"

# Save the model weights
torch.save(best_model.state_dict(), weights_path)