from utils import linear_warmup_cosine_decay, get_cycles_buildoff, lr_search
import pytorch_lightning as pl
import torch
import torchvision
import torchmetrics
import os
import random
from itertools import chain
from diffusers import AutoencoderKL
from torch import nn, einsum
import torch.nn.functional as F

class Classifire(pl.LightningModule):
    def __init__(
        self,
        steps,
        input_shape,
        learning_rate=1e-4,
        weight_decay=0.0001,
        dropout_p=0.0,
        linear_warmup_ratio=0.01,
        **_,
    ):
        super().__init__()
        self.save_hyperparameters() 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps
        self.linear_warmup_ratio = linear_warmup_ratio
        self.criterion = torch.nn.NLLLoss()
        self.init_model(input_shape, dropout_p)
    
    def init_model(self, input_shape, dropout_p):
        feature_layers = [
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_p)
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout(p=dropout_p)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_p)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout(p=dropout_p)
            )
        ]
        self.features = nn.Sequential(*feature_layers)
        n_sizes = self._get_conv_output(input_shape)
        output_layers = [
            nn.Linear(n_sizes, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        ]
        self.accuracy = torchmetrics.Accuracy()
        self.output = nn.Sequential(*output_layers)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.features(input)
        print("output_feat", output_feat.shape)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        print("Conv length:", n_size)
        return n_size
        
    def forward(self, x):
        x_l = x[:, 0, ...]
        x_r = x[:, 1, ...]
        x_l = self.features(x_l).view(x.size(0), -1)
        x_r = self.features(x_r).view(x.size(0), -1)
        linspace = torch.linspace(0, 1, x_l.shape[1], device = self.device)
        x_l = x_l * torch.sin(linspace)
        x_r = x_r * torch.cos(linspace)
        x = torch.maximum(x_l, x_r)
        x = self.output(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
        warmup_steps = int(self.linear_warmup_ratio * self.steps)
        scheduler = {
            "scheduler": linear_warmup_cosine_decay(optimizer, warmup_steps, self.steps),
            "interval": "step",
        }
        # scheduler = {
        #     "scheduler": get_cycles_buildoff(
        #         optimizer, 
        #         num_warmup_steps = warmup_steps, 
        #         num_training_steps = self.steps, 
        #         noise_amount = 0.005,
        #         num_cycles = 100,
        #         merge_cycles = 10,
        #         last_epoch = -1
        #     ),
        #     "interval": "step",
        # }
        # scheduler = {
        #     "scheduler": lr_search(optimizer),
        #     "interval": "step",
        # }
        return [optimizer], [scheduler]

    def shot(self, batch, name, image_logging = False):
        image_grid, target = batch
        logits = self.forward(image_grid)
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)
        self.log(f"{name}_loss", loss, on_step=True, on_epoch=True, logger=True)        
        self.log(f"{name}_acc", acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shot(batch, "train", image_logging = True)

    def validation_step(self, batch, batch_idx):
        return self.shot(batch, "val")

    def log_sources(self, files):
        output_folder = os.path.join(self.logger.log_dir, "source_during_training")
        os.makedirs(output_folder, exist_ok=True)
        for file in files:
            with open(file, 'r') as input:
                source = input.read()
                output_path = os.path.join(output_folder, file)
                with open(output_path, 'w') as output:
                    output.write(source)

    def on_train_start(self):
        self.log_sources(['model.py'])
        