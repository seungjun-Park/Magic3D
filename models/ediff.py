import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor, T5Tokenizer

class EDiff(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_encoder = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_image_encoder = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.t5_eoncder = T5Tokenizer.from_pretrained("t5-large")

    def loss(self):
        return

    def forward(self, image, text):
        return

    def training_step(self, batch, batch_idx):
        image, text = batch
        output = self(image, text)

        return

    def validation_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        return

