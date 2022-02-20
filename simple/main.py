import sys
import math
import torch
import random
import argparse

from torch import nn
from torch import optim
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter


class SelfAttention(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.k = nn.Linear(n_embed, n_embed)
        self.q = nn.Linear(n_embed, n_embed)
        self.v = nn.Linear(n_embed, n_embed)

        self.ln = nn.LayerNorm(n_embed, n_embed)
        self.mlp = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        # batch_size = 10, word_length = 5, embed_size = 10
        # meaning x.size() == [10, 5, 10]

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        y_hat = (q @ k.transpose(2, 1)) * (1.0 / math.sqrt(k.size(-1)))
        y_hat = F.softmax(y_hat, dim=-1)
        print(y_hat)
        y_hat = y_hat @ v


        return self.ln(x + y_hat)


class Model(nn.Module):
    def __init__(self, vocab, n_embed, n_output):
        super().__init__()

        self.embedding = nn.Embedding(len(vocab), n_embed)
        self.attn = SelfAttention(n_embed)
        self.ln = nn.LayerNorm(n_output, n_output)
        self.decoder = nn.Linear(n_embed * len(vocab), n_output)

    def get_embedding_for_word(self, word):
        idx = torch.tensor([vocab[word]])
        return self.embedding(idx)

    def get_embedding_for_sentence(self, sentence):
        t = torch.tensor([])
        for word in sentence.split(" "):
            t = torch.cat((t, self.get_embedding_for_word(word)))
        return t

    def forward(self, batch_x):
        x_embed = torch.stack([self.get_embedding_for_sentence(x) for x in batch_x])
        x_attn  = self.attn(x_embed)
        y_proj  = self.decoder(x_attn.view(-1, x_attn.size(1) * x_attn.size(2)))

        return y_proj


# The goal of this counting transformer is to convert
# An input sequence such as: "a b c c a"
# To an output sequence [2, 1, 2, 0, 0]
# Where the values are the counts for the characters at the index in the vocab.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple attention")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    vocab = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "f": 5,
        "g": 6,
        "h": 7,
        "i": 8,
        "j": 9
    }

    n_embed = 10
    n_vocab = len(vocab)
    model = Model(vocab, n_embed, n_vocab)

    if args.mode == "eval":
        if args.input is None:
            print("eval mode expected input")
            sys.exit(1)

        ckpt = torch.load("./saves/good2.pt")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        y = model([args.input])
        print(y)

    if args.mode == "train":
        # Generate some random training data.
        n_training_samples = 15000
        batch_size = 64
        vals = list(vocab.keys())

        training_data = []
        for i in range(n_training_samples // batch_size):
            batch = []
            for bi in range(batch_size):
                chars = []
                for j in range(len(vocab)):
                    chars.append(random.choice(vals))

                x = " ".join(chars)
                y = [chars.count(c) for c in vals]

                # We are going to use this in eval.
                if x == "a b c d e f g h i j":
                    x = "a a b b c c d d e e"

                batch.append({
                    "x": x,
                    "y": y
                })

            training_data.append(batch)

        # We now have a long list of training data samples that look like:
        # e.g {'x': 'a a e b e', 'y': [2, 1, 0, 0, 2]}

        writer = SummaryWriter("runs/test-6")
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        epochs = 200
        log_every_n = 1000
        for epoch in range(epochs):
            for (i, batch) in enumerate(training_data):
                batch_x = [b["x"] for b in batch]
                batch_y = torch.stack([torch.tensor(b["y"]) for b in batch])

                optimizer.zero_grad()

                y_hat = model(batch_x)
                loss = F.l1_loss(y_hat, batch_y)

                writer.add_scalar("loss", loss, epoch * n_training_samples + i)

                if i % log_every_n == 0:
                    print(loss)

                loss.backward()
                optimizer.step()

        writer.close()
        torch.save({"model_state_dict": model.state_dict()}, "./saves/good2.pt")


# Things to play around with:
#
# - batch inputs (DONE)
# - layer norm
# - visualize the attention
# - cross entropy loss
# - variable output size
# - >1 head
