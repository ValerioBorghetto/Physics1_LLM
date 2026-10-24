from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
import torch

from data_loader import create_dataloader_v1
from llm import GPTModel
from training import train_model_simple, plot_losses


def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    ##############################
    # Load local training data
    ##############################

    file_path = Path(__file__).resolve().parent / "data" / "physics1_book" / "physics1_expanded_train.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # Initialize custom GPT model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"]
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"]
    )

    ##############################
    # Train custom GPT model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=200, eval_iter=10,
        start_context="A physical model", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    
    GPT_CONFIG = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 192,
        "n_heads": 4,
        "n_layers": 4,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    OTHER_SETTINGS = {
        "learning_rate": 3e-4,
        "num_epochs": 15,
        "batch_size": 8,
        "weight_decay": 0.01
    }


    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    fig = plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    fig.savefig("loss.pdf")
    plt.close(fig)

    torch.save(model.state_dict(), "model_physics1.pth")
    model = GPTModel(GPT_CONFIG)
    model.load_state_dict(torch.load("model_physics1.pth", weights_only=True, map_location="cpu"))
