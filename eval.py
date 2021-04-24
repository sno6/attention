from dataset import RecipeDataset
from model import RecipeModel
import torch

if __name__ == '__main__':
    batch_size = 8
    n_embed = 512
    n_hidden = 512

    dataset = RecipeDataset("./data/training.json")
    
    model = RecipeModel.load_from_checkpoint(
        checkpoint_path='./checkpoints/recipe-checkpoint-epoch=00-val_join_loss=0.17.ckpt',
        enc_vocab_size=len(dataset.enc_vocab),
        dec_quantity_vocab_size=len(dataset.dec_quantity_vocab),
        dec_unit_vocab_size=len(dataset.dec_unit_vocab),
        n_embed=n_embed,
        n_hidden=n_hidden
    )
    model = model.eval()

    while True:
        inp = input(">>> ")
        s_vec = torch.tensor(dataset.encode_sentence(inp)).float()
        q, u = model(s_vec.unsqueeze(0))

        print(f"Quantity: {dataset.decode_quantity(q)}")
        print(f"Unit: {dataset.decode_unit(u)}")


