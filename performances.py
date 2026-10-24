
import torch
import tiktoken

from llm import GPTModel, generate_text_simple


# Stessa configurazione usata in pretraining e fine-tuning
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 192,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Carica il modello finetuned
model = GPTModel(GPT_CONFIG)

state_dict = torch.load(
    "model_finetuned_physics1.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 2. Carica lo stesso tokenizer del training
tokenizer = tiktoken.get_encoding("gpt2")


# 3. Funzione di inference
def generate_response(instruction):
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
        "\n\n### Response:\n"
    )

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded], device=device)

    if input_ids.shape[1] >= GPT_CONFIG["context_length"]:
        raise ValueError("Prompt troppo lungo per il modello")

    max_new_tokens = min(
        150,
        GPT_CONFIG["context_length"] - input_ids.shape[1]
    )

    with torch.inference_mode():
        output_ids = generate_text_simple(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=GPT_CONFIG["context_length"]
        )

    # Decodifica solo i token generati
    generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
    response = tokenizer.decode(generated_ids)

    # Interrompe la risposta se compare un separatore
    response = response.split("### Instruction:")[0]
    response = response.split("### Response:")[0]
    response = response.split("<|endoftext|>")[0]

    return response.strip()


if __name__ == "__main__":
    while True:
        instruction = input("\nYou: ")

        if instruction.lower() in ("exit", "quit"):
            break

        response = generate_response(instruction)
        print(f"\nModel: {response}")
