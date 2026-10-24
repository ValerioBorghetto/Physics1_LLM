"""Response-only supervised fine-tuning for the user's custom GPTModel.
Place beside llm.py, model.pth and the physics_sft_*.json files.

SECONDA VERSIONE DEL FINETUNING, NON REVISIONATA.

Example: python finetuning_improved.py --epochs 4 --batch-size 4 --grad-accum 4
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from llm import GPTModel

ROOT=Path(__file__).resolve().parent
CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 192,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False
}

EOS=50256


def prompt_for(item):
    question=item['instruction'].strip()
    extra=item.get('input','').strip()
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{question}"
            + (f"\n\n### Input:\n{extra}" if extra else "")
            + "\n\n### Response:\n")


class ResponseDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        items=json.loads(Path(path).read_text(encoding='utf-8'))
        self.examples=[]
        self.skipped=0
        for item in items:
            response=item.get('response',item.get('output'))
            if not isinstance(response,str) or not response.strip():
                self.skipped+=1; continue
            prompt_ids=tokenizer.encode(prompt_for(item),allowed_special=set())
            response_ids=tokenizer.encode(response.strip(),allowed_special=set())+[EOS]
            # Preserve the entire instruction: skip overlong cases rather than silently
            # truncating their response and training on nonsensical targets.
            if len(prompt_ids)+len(response_ids)>max_length or len(prompt_ids)<1:
                self.skipped+=1; continue
            ids=prompt_ids+response_ids
            input_ids=ids[:-1]
            labels=ids[1:]
            # labels[j] predicts ids[j+1]. Only predict response (and EOS).
            labels[:len(prompt_ids)-1]=[-100]*(len(prompt_ids)-1)
            self.examples.append((input_ids,labels))
        if not self.examples:
            raise RuntimeError(f'No valid examples in {path} for context length {max_length}')
        print(f'{Path(path).name}: {len(self.examples)} retained, {self.skipped} skipped')

    def __len__(self):return len(self.examples)
    def __getitem__(self,idx):return self.examples[idx]


def collate(batch):
    max_len=max(len(x[0]) for x in batch)
    xs=torch.full((len(batch),max_len),EOS,dtype=torch.long)
    ys=torch.full((len(batch),max_len),-100,dtype=torch.long)
    for i,(x,y) in enumerate(batch):
        xs[i,:len(x)]=torch.tensor(x)
        ys[i,:len(y)]=torch.tensor(y)
    return xs,ys


def mean_loss(model,loader,device):
    model.eval()
    total=0.0; n=0
    with torch.inference_mode():
        for xs,ys in loader:
            xs=xs.to(device);ys=ys.to(device)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(xs)
                total+=float(F.cross_entropy(logits.flatten(0,1),ys.flatten(),ignore_index=-100,reduction='sum'))
            n+=int((ys!=-100).sum())
    return total/max(n,1)


def main(args):
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=tiktoken.get_encoding('gpt2')
    train=ResponseDataset(ROOT/'data/physics1_sft_train.json',tok,CONFIG['context_length'])
    val=ResponseDataset(ROOT/'data/physics1_sft_val.json',tok,CONFIG['context_length'])
    test=ResponseDataset(ROOT/'data/physics1_sft_test.json',tok,CONFIG['context_length'])
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,collate_fn=collate)
    test_loader=DataLoader(test,batch_size=args.batch_size,shuffle=False,collate_fn=collate)
    model=GPTModel(CONFIG)
    # This architecture MUST match the one used during pretraining.
    weights=torch.load(ROOT/'model_physics1.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True)
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda'))
    best=float('inf')
    print('Device:',device,'Parameters:',sum(p.numel() for p in model.parameters()))
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for i,(xs,ys) in enumerate(train_loader):
            xs=xs.to(device);ys=ys.to(device)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(xs)
                loss=F.cross_entropy(logits.flatten(0,1),ys.flatten(),ignore_index=-100)
            # Account for a shorter final accumulation step.
            window_start=(i//args.grad_accum)*args.grad_accum
            divisor=min(args.grad_accum,len(train_loader)-window_start)
            scaler.scale(loss/divisor).backward()
            if (i+1)%args.grad_accum==0 or i+1==len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        vl=mean_loss(model,val_loader,device)
        print(f'Epoch {epoch+1}/{args.epochs}: validation response-token loss={vl:.4f}')
        if vl<best:
            best=vl
            torch.save(model.state_dict(),ROOT/'model_finetuned_physics1.pth')
    model.load_state_dict(torch.load(ROOT/'model_finetuned_physics1.pth',map_location=device,weights_only=True))
    print(f'Best validation loss={best:.4f}; held-out test loss={mean_loss(model,test_loader,device):.4f}')
    print('Saved best checkpoint: model_finetuned_physics1.pth')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--grad-accum',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--weight-decay',type=float,default=0.05)
    p.add_argument('--seed',type=int,default=123)
    main(p.parse_args())
