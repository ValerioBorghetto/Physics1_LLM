from importlib.metadata import version


print("Torch version", version("torch"))
print("Tiktoken version", version("tiktoken"))

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

max_length=10 #lunghezza dei chunk
stride=10#di quanto scorre ogni batch
batch_size=8 #dimensione di ogni batch
vocab_size_gpt2=50257 #dinemnsione del vocabolario di input
vocab_size_custom=52000
context_lenght=1024#per il positional encoding
inner_dim=257#dimensione dell'embedding

#Mi serve una classe che divida il testo in chunk e consideri lo slide, per avere chunk di input e di output

class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_txt_tok = []
        self.target_txt_tok = []

        token_txt = tokenizer.encode(txt) #tokenizza il testo
        if hasattr(token_txt, "ids"):
            token_txt = token_txt.ids

        #loop per definirie i chunkl con lo stride
        for i in range(0, len(token_txt)-max_length, stride):
            input_chunk = token_txt[i:i+max_length]
            output_chunk = token_txt[i+1:max_length+1+i]
            self.input_txt_tok.append(torch.tensor(input_chunk))
            self.target_txt_tok.append(torch.tensor(output_chunk))
    #metodi necessari per far capire a pytorch che è una classe indicizzabile
    def __len__(self):
        return len(self.input_txt_tok)

    def __getitem__(self, idx):
        return self.input_txt_tok[idx], self.target_txt_tok[idx]

#Ora, grazie alla classe che tokenizza e divide il testo, posso creare il DataLoader
def create_dataloader_v1(txt, batch_size, max_length, stride):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = LLMDataset(txt, tokenizer, max_length, stride)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
"""
def create_dataloader_custom(txt, batch_size, max_length, stride):
    from tokenizers import ByteLevelBPETokenizer #funzione di huggingface
    tokenizer=ByteLevelBPETokenizer()
    tokenizer.train(files="data/physics_pretraining.txt", vocab_size=vocab_size_custom, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
    tokenizer.save_model("PhysicsBPE")
    dataset=LLMDataset(txt, tokenizer, max_length, stride)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
"""    

