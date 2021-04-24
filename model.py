import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import RecipeDataset
import pytorch_lightning as pl
import math

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class Attention(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()

        self.queries = nn.Linear(n_embed, n_embed)
        self.keys = nn.Linear(n_embed, n_embed)
        self.values = nn.Linear(n_embed, n_embed)

    # embedded.shape = [B, max_sentence_length, n_embed]
    def forward(self, embedded):
        q = self.queries(embedded)
        k = self.keys(embedded)
        v = self.values(embedded)

        attn_weights = F.softmax(q @ k.transpose(1, 2)
                                 * (1.0 / math.sqrt(k.size(-1))), 2)

        attn = torch.matmul(attn_weights, v)

        return embedded + attn


class RecipeModel(pl.LightningModule):
    def __init__(self, enc_vocab_size: int, dec_quantity_vocab_size: int, dec_unit_vocab_size: int, n_embed: int, n_hidden: int):
        super().__init__()

        self.enc_vocab_size = enc_vocab_size
        self.dec_quantity_vocab_size = dec_quantity_vocab_size
        self.dec_unit_vocab_size = dec_unit_vocab_size

        self.n_embed = n_embed
        self.n_hidden = n_hidden

        self.embedding = nn.Linear(self.enc_vocab_size, self.n_embed)
        self.attention = Attention(self.n_embed)

        self.q_head = nn.Linear(self.n_embed, self.dec_quantity_vocab_size)
        self.u_head = nn.Linear(self.n_embed, self.dec_unit_vocab_size)

    # x.shape = [B, max_sentence_length, enc_vocab_size]
    def forward(self, x):
        # Pass our one-hot encoded input sentence through an embedding layer.
        # This should build a better representation of our one-hot sequence.
        o = self.embedding(x)
        o = F.relu(o)

        # Extract context from word relationships in the sentence.
        o = self.attention(o)

        # Sum all word vectors to reduce dimensionality. Intuitively, words that don't contribute
        # much to the output will sum to a small value.
        o = o.sum(1)

        q_out = self.q_head(o)
        u_out = self.u_head(o)

        return q_out, u_out

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']

        q_out, u_out = self.forward(x)

        q_loss = F.cross_entropy(q_out, torch.tensor(
            [y_h.argmax() for y_h in y['quantity']]))

        u_loss = F.cross_entropy(u_out, torch.tensor(
            [y_h.argmax() for y_h in y['unit']]))

        joint_loss = q_loss + u_loss

        self.log("train_q_loss", q_loss)
        self.log("train_u_loss", u_loss)
        self.log("train_join_loss", joint_loss)

        return joint_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']

        q_out, u_out = self.forward(x)

        q_loss = F.cross_entropy(q_out, torch.tensor(
            [y_h.argmax() for y_h in y['quantity']]))

        u_loss = F.cross_entropy(u_out, torch.tensor(
            [y_h.argmax() for y_h in y['unit']]))

        joint_loss = q_loss + u_loss

        self.log("val_q_loss", q_loss)
        self.log("val_u_loss", u_loss)
        self.log("val_join_loss", joint_loss)

        return joint_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    batch_size = 8
    n_embed = 512
    n_hidden = 512

    checkpoint_callback = ModelCheckpoint(
        monitor='val_join_loss',
        dirpath='./checkpoints',
        filename='recipe-checkpoint-{epoch:02d}-{val_join_loss:.2f}'
    )

    dataset = RecipeDataset("./data/training.json")
    train, val = random_split(dataset, [80000, 20000])

    model = RecipeModel(
        len(dataset.enc_vocab),
        len(dataset.dec_quantity_vocab),
        len(dataset.dec_unit_vocab),
        n_embed,
        n_hidden
    )

    trainer = pl.Trainer(max_epochs=5, gpus=0, callbacks=[
                         checkpoint_callback], logger=None)

    trainer.fit(model, DataLoader(train, batch_size=batch_size),
                DataLoader(val, batch_size=batch_size))
