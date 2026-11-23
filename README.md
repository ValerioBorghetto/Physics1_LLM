# Pre-training and Finetuning from zero a Physics-based LLM

The goal is to pre-train and fine-tuning from zero a small LLM-based architecture, training it on physics-based data. 
**How do sparse, structured Physics-based data differ from NLP data?**
In the current implementation, the parameters and baseline setup are the following:

## Current Architecture & Training Setup

| Component / Parameter | Current Specification |
| :--- | :--- |
| **Architecture Type** | Dense Decoder-only Transformer  |
| **Model Size** | `~21.1M` parameters (`~19.3M` in token embeddings & LM head, `~1.8M` in Transformer blocks) |
| **Layers / Heads / Hidden Dim** | `4` layers, `4` heads (`d_head = 48`), `d_model = 192` |
| **Context Window** | `256` tokens |
| **Pre-training Dataset** | Custom Physics 1 corpus |
| **Fine-tuning Dataset** | Instruction Q&A |
| **Input Data Modality** | Text-only (natural language & ASCII mathematical formulas) |
| **Tokenizer** | `tiktoken` GPT-2 BPE tokenizer (`vocab_size = 50,257`) |
| **Precision & Training** | Pre-training: `FP32`, `AdamW`; SFT: `FP16` mixed precision, `AdamW` (`lr=3e-5`) |

## Work in Progress

The current baseline relies on unstructured text and a dense transformer backbone. Ongoing and planned developments focus on adapting the architecture to handle structured scientific representations and scaling efficiency:

- Extending the input pipeline and tokenization/encoding schemes beyond plain text to natively ingest structured physical data, including numerical data and graph representations
- Replacing standard dense Feed-Forward Network (FFN) layers with **Mixture-of-Experts (MoE)** layers to increase model capacity and domain specialization. 
