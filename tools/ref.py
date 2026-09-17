from pathlib import Path
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

PROMPT = "Hello, I'm a language model,"
OUT = Path(__file__).resolve().parent.parent / "models" / "ref"

def dump(name: str, t: torch.Tensor):
    a = t.detach().squeeze(0).contiguous().numpy()
    a.tofile(OUT / f"{name}.bin")
    print(f"{name:8s} {str(a.shape):14s} {a.dtype}")

def hook(name: str, module: torch.nn.Module):
    module.register_forward_hook(lambda mod, inp, out: dump(name, out))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    hook("ln1_00", model.transformer.h[0].ln_1)
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    ids = tokenizer(PROMPT, return_tensors="pt").input_ids # shape: [1, T]
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    dump("tokens", ids.to(torch.int32))
    for i, h in enumerate(out.hidden_states):
        dump(f"h{i:02d}", h)
    dump("logits", out.logits)

    print("ids:", ids[0].tolist())
    print("next:", out.logits[0, -1].argmax().item(), repr(tokenizer.decode(out.logits[0, -1].argmax())))

if __name__=="__main__":
    main()
