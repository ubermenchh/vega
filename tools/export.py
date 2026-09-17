import shutil
import struct
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel

MAGIC = 0x47505432
OUT = Path(__file__).resolve().parent.parent / "models"

def main() -> None:
    OUT.mkdir(exist_ok=True)
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    sd = model.state_dict()
    cfg = model.config

    # HF Conv1D stores (in, out); we want (out, in) so each output row is contiguous
    tensors = [sd["transformer.wte.weight"], sd["transformer.wpe.weight"]]
    for i in range(cfg.n_layer):
        p = f"transformer.h.{i}."
        tensors += [
            sd[p + "ln_1.weight"], sd[p + "ln_1.bias"],
            sd[p + "attn.c_attn.weight"].T, sd[p + "attn.c_attn.bias"],
            sd[p + "attn.c_proj.weight"].T, sd[p + "attn.c_proj.bias"],
            sd[p + "ln_2.weight"], sd[p + "ln_2.bias"],
            sd[p + "mlp.c_fc.weight"].T, sd[p + "mlp.c_fc.bias"],
            sd[p + "mlp.c_proj.weight"].T, sd[p + "mlp.c_proj.bias"],
        ]
    tensors += [sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"]]

    path = OUT / "gpt2-124M.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<8i", MAGIC, 1, cfg.n_layer, cfg.n_head, cfg.n_embd,
            cfg.vocab_size, cfg.n_positions, 0))
        f.writelines(t.detach().contiguous().float().numpy().tobytes() for t in tensors)

    n_params = sum(t.numel() for t in tensors)
    size = path.stat().st_size
    assert size == n_params * 4 + 32, (size, n_params)
    print(f"wrote {path}: {size} bytes, {n_params} params")

    for name in ("vocab.json", "merges.txt"):
        shutil.copyfile(hf_hub_download("openai-community/gpt2", name), OUT / name)
        print(f"wrote {OUT / name}")


if __name__=="__main__":
    main()
