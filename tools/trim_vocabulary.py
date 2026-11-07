"""Specialize the Apache Index GGUF vocabulary for offline English–Cebuano."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import unicodedata

from gguf import GGUFReader, GGUFWriter
import numpy as np
from tokenizers import Tokenizer


def byte_decoder():
    # Inverse of the byte alphabet used by GPT-style byte-level BPE.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [byte for byte in range(256) if byte not in visible]
    return dict(zip(map(chr, visible + list(range(256, 256 + len(missing)))), visible + missing))


def latin_token(piece, decoder):
    try:
        text = bytes(decoder[ch] for ch in piece).decode("utf-8")
    except (UnicodeDecodeError, KeyError):
        return False
    return bool(text) and all(
        ord(ch) < 128 or unicodedata.category(ch)[0] in "PMNZ"
        or unicodedata.category(ch) == "Sc" or unicodedata.name(ch, "").startswith("LATIN ")
        or ch in "°µ"
        for ch in text
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--size", type=int, default=55000)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        raise ValueError("Use a separate output file")

    source = GGUFReader(args.source)
    fields = source.fields
    vocab_key = "tokenizer.ggml.tokens"
    types_key = "tokenizer.ggml.token_type"
    merges_key = "tokenizer.ggml.merges"
    tokens = fields[vocab_key].contents()
    types = fields[types_key].contents()
    merges = fields[merges_key].contents()
    lookup = {piece: i for i, piece in enumerate(tokens)}
    if len(tokens) != len(lookup) or len(tokens) != len(types):
        raise ValueError("Unexpected vocabulary format")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if fields["general.architecture"].contents() != "qwen35":
        raise ValueError("Only the pinned qwen35 base is supported")
    decoder = byte_decoder()
    pieces = defaultdict(set)
    edges = []
    for merge in merges:
        left, right = merge.split(" ", 1)
        if left not in lookup or right not in lookup or left + right not in lookup:
            continue
        a, b, final = lookup[left], lookup[right], lookup[left + right]
        pieces[final].update((a, b))
        edges.append((merge, a, b, final))

    essential = {i for i, (p, t) in enumerate(zip(tokens, types)) if t != 1 or p in decoder}
    header = "<|im_start|>user\n请将以下文本翻译为宿务语，直接输出翻译结果，不要进行任何解释。\n\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    originals = [header]
    for line in args.pairs.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        originals.extend([row["en"], row["ceb"]])
    for encoded in tokenizer.encode_batch(originals, add_special_tokens=False):
        for ident, piece in zip(encoded.ids, encoded.tokens):
            if tokens[ident] != piece:
                raise ValueError("The tokenizer and GGUF token tables do not match")
            essential.add(ident)

    retained = set()

    def closure(roots):
        found = set()
        todo = list(roots)
        while todo:
            item = todo.pop()
            if item not in retained and item not in found:
                found.add(item)
                todo.extend(pieces.get(item, ()))
        return found

    retained.update(closure(essential))
    if len(retained) > args.size:
        raise ValueError(f"Required token closure already has {len(retained)} tokens")
    for ident, piece in enumerate(tokens):
        if len(retained) == args.size:
            break
        if ident not in retained and latin_token(piece, decoder):
            additions = closure((ident,))
            if len(retained) + len(additions) <= args.size:
                retained.update(additions)
    chosen = sorted(retained)
    remapped = {old: new for new, old in enumerate(chosen)}
    new_merges = [merge for merge, a, b, final in edges if {a, b, final} <= retained]
    print(f"Keeping {len(chosen)} of {len(tokens)} tokens and {len(new_merges)} merges", flush=True)

    writer = GGUFWriter(args.output, "qwen35")
    for name, field in fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        if name == vocab_key:
            value = [tokens[i] for i in chosen]
        elif name == types_key:
            value = [types[i] for i in chosen]
        elif name == merges_key:
            value = new_merges
        elif name.startswith("tokenizer.ggml.") and name.endswith("_token_id"):
            value = remapped[field.contents()]
        elif name == "general.name":
            value = "QnVoice English to Cebuano compact Index base"
        elif name == "general.size_label":
            value = "1.5B"
        else:
            value = field.contents()
        writer.add_key_value(name, value, field.types[0], field.types[1] if len(field.types) > 1 else None)
    for tensor in source.tensors:
        data = tensor.data
        if tensor.name in ("token_embd.weight", "output.weight"):
            if data.shape[0] != len(tokens):
                raise ValueError(f"Unexpected {tensor.name} shape")
            data = np.ascontiguousarray(data[chosen])
        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(args.output, args.output.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()
