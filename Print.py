import torch
from SegFormer import SegFormer
import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
from snntorch import utils
from DataLoader_copy import GetData


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
        # self.criterion = SF.ce_rate_loss()
        # self.metrics = accuracy_calc()

    def custom_forward_pass(self, data, num_steps= 5):
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
        # torch.cuda.empty_cache()
        train_image, train_segment = batch
        train_loss, train_accuracy = self.process(train_image, train_segment)
        # torch.cuda.empty_cache()
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss



    def validation_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        val_image, val_segment = batch
        val_loss, val_accuracy = self.process(val_image, val_segment)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # test_acc = self.batch_accuracy(test_loader, net, num_steps)
        return val_loss


model = SegFormerModel()
datamodule = GetData(batch_size=2)

print("120")
checkPointPath = "./checkpoints/last.ckpt"

best_model = SegFormerModel.load_from_checkpoint(checkPointPath)
best_model.eval()
print("121")

# datamodule.setup()
# val_loader = datamodule.val_dataloader()
# print("122")

# pred_masks = []
# gt_masks = []
# val_images = []

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for idx, val_batch in enumerate(val_loader):
#     if idx == 5:
#         break
#     with torch.no_grad():
#         print("yy")
#         val_image, val_segment = val_batch
#         val_image = val_image.to(device)
#         pred_mask = best_model(val_image).argmax(dim=1).cpu().numpy()
#         pred_masks.append(pred_mask)
#         gt_masks.append(val_segment)
#         val_images.append(val_image.to(device))


# # %%
# print(gt_masks[0].shape)
# print(pred_masks[0].shape)
# print(val_images[0].shape)

# import os
# temp_folder = 'testing'

# import matplotlib.pyplot as plt
# from PIL import Image
# os.makedirs(temp_folder, exist_ok=True)

# # Visualizing predicted masks
# num_samples_to_visualize = 5  # Choose the number of samples to visualize
# for i in range(num_samples_to_visualize):
#     plt.figure(figsize=(12, 8))

#     # Plot original image
#     plt.subplot(1, 3, 1)
#     original_img = val_images[i][0].cpu().permute(1, 2, 0).numpy()
#     plt.imshow(original_img)
#     plt.title("original_img")

#     # Plot ground truth
#     plt.subplot(1, 3, 2)
#     ground_truth = gt_masks[i][0].cpu().numpy()  # Assuming gt_masks is a tensor
#     plt.imshow(ground_truth)
#     plt.title('ground truth')

#     # Plot predicted mask
#     plt.subplot(1, 3, 3)
#     predicted_mask = torch.tensor(pred_masks[i][0]).cpu().numpy()  # Assuming pred_masks is a numpy array
#     plt.imshow(predicted_mask)
#     plt.title('Predicted Mask')

#     # Save the figure
#     plt.savefig(os.path.join(temp_folder, f'image_{i + 1}.png'))
