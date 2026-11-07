"""Convert the tested MLX adapter tensors to llama.cpp GGUF without copying the base."""
import argparse
import json
from pathlib import Path
import re

import gguf
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    settings = json.loads((args.adapter / "adapter_config.json").read_text())
    params = settings["lora_parameters"]
    mapping = {"gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}
    writer = gguf.GGUFWriter(args.output, "qwen35")
    writer.add_type("adapter")
    writer.add_name("QnVoice English to Cebuano Conversation Adapter")
    writer.add_string("adapter.type", "lora")
    # MLX multiplies A @ B by scale; llama.cpp uses alpha / rank instead.
    if not 0 < args.strength <= 1:
        raise ValueError("Adapter strength must be between zero and one")
    writer.add_float32("adapter.lora.alpha", params["rank"] * params["scale"] * args.strength)
    pattern = re.compile(r"(?:language_model\.)?model\.layers\.(\d+)\.mlp\.(\w+)\.lora_([ab])")
    with safe_open(args.weights or args.adapter / "adapters.safetensors", framework="pt", device="cpu") as weights:
        expected = settings["num_layers"] * len(params["keys"]) * 2
        if len(weights.keys()) != expected:
            raise ValueError(f"Expected {expected} trained tensors, found {len(weights.keys())}")
        for key in weights.keys():
            match = pattern.fullmatch(key)
            if not match:
                raise ValueError(f"Unexpected LoRA tensor {key}")
            layer, projection, part = match.groups()
            tensor = weights.get_tensor(key).T.contiguous().half().numpy()
            if (tensor.shape[0] if part == "a" else tensor.shape[1]) != params["rank"]:
                raise ValueError(f"Unexpected rank in {key}")
            writer.add_tensor(f"blk.{layer}.{mapping[projection]}.weight.lora_{part}", tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(args.output, args.output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
