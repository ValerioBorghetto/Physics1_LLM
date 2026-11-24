import tiktoken
import torch
import torch.nn as nn
from data_loader import create_dataloader_v1#, create_dataloader_custom
"""
#Via più facile, semplicemente prendo un modello da huggingface già pronto
from transformers import GPT2Config, GPT2LMHeadModel
cfg = GPT2Config(
    vocab_size=32000,
    n_positions=1024,
    n_ctx=1024,
    n_embd=1024,
    n_layer=24,
    n_head=16,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)
#model = GPT2LMHeadModel(cfg)
"""
#Via con pytorch invece, tutto custom
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #embedding
        d = cfg["emb_dim"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.pos_emb = nn.Embedding(cfg["context_length"], d)
        self.drop = nn.Dropout(cfg["drop_rate"])
        #transformer
        transf_layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg["n_heads"], dim_feedforward=4*d, dropout=cfg["drop_rate"], activation="gelu", batch_first=True, norm_first=True)
        self.trf_blocks=nn.TransformerEncoder(transf_layer, num_layers=cfg["n_layers"], enable_nested_tensor=False)
        #norma e espansione finale
        self.norm=nn.LayerNorm(d)
        self.out_head = nn.Linear(d, cfg["vocab_size"], bias=False)

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        tok_embeds = self.tok_emb(idx)
        positions = torch.arange(seq_len, device=idx.device)
        pos_embeds = self.pos_emb(positions)
        x = tok_embeds + pos_embeds # Shape [batch_size, num_tokens, emb_size]
        x=self.drop(x)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=idx.device, dtype=torch.bool), diagonal=1)
        x=self.trf_blocks(x, mask=causal_mask)
        x=self.norm(x)
        logits = self.out_head(x)
        return logits

#Ora una funzione che generi il testo
def generate_text_simple(model, idx, max_new_tokens, context_size):
    #idx è il testo indicizzato (tokenizzato e embeddato), quindi ha dimensione [batch_size, emb_dim]
    for _ in range(max_new_tokens):
        #taglia l'idx se ha troppi token rispetto a quelli supportati dall'LLM
        idx_cut = idx[:, -context_size:]
        with torch.no_grad():
            logits=model(idx_cut) #conterrà la predizione per tutti i token, voglio solo l'ultimo...(batch, n_token, vocab_size) becomes (batch, vocab_size)
            logits = logits[:,-1, :]
            #prendi per ogni batch l'indice del valore col valore più alto...qui ancora non si è convertito il logits con la probabilità
            #idx_next=torch.argmax(logits, dim=-1, keepdim=True)
            #nel modo seguente invece usi la temperatura, in modo che la generazione sia ocmunque stocastica
            probs = torch.softmax(logits/temperature, dim=-1)
            idx_next=torch.multimodal(probs, num_samples=1)
            #appendi l'indice alla seuqenza
            idx=torch.cat((idx, idx_next), dim=1) #(batch_size, n_tokens+1)

    return idx

#main per testare lo script, ancora non allenato
def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()



        
        

        


