import matplotlib.pyplot as plt
import os
import tiktoken
import torch

from data_loader import create_dataloader_v1#, create_dataloader_custom
from llm import GPTModel, generate_text_simple

#codificare il testo
def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)# add batch dimension
    return encoded_tensor

def token_to_text(token, tokenizer):
    flat=token.squeeze(0)# remove batch dimension
    return tokenizer.decode(flat.tolist())

#calcolare la loss per ogni batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits=model(input_batch)
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

#calcolare la loss totale del loader completo
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss=0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches=len(data_loader)
    else:
        num_batches=min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader): #la i è il numero di batch
        if i< num_batches:
            loss=calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size=model.pos_emb.weight.shape[0]
    encoded=text_to_token(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids=generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
        decoded_text=token_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss=calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_iter, start_context, tokenizer, eval_freq = 1):
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen = 0
    global_step = -1

    #main train loop
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss=calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1
            #Step di valutazione (opzionale). la loss appesa solo qua
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        #printa un testo di sample dopo ogni epoca
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_token_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    return fig

def main(gpt_config, settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    
    
    
        
