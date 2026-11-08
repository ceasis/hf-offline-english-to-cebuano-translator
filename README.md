# QnVoice Offline English-to-Cebuano Translator

QnVoice's research release combines a vocabulary-reduced Index-Translate-2B model with a rank-16 conversation adapter trained for English-to-Cebuano translation. Both GGUF files run locally. The base is 978 MB and the adapter is 263 KB. [Pinned Hugging Face release](https://huggingface.co/CholoAsis/qnvoice-cebuano-translation/tree/27efd4da6be61a51219234ca584586cb084a6133)

## Results

The model scored higher than the Helsinki-NLP OPUS-MT baseline on FLORES-200 and the 40 QnVoice conversation examples. Helsinki scored higher on Tatoeba. Higher chrF++ and BLEU indicate closer agreement with the reference translations.

| Held-out evaluation | Sentences | Helsinki chrF++ | QnVoice chrF++ | Helsinki BLEU | QnVoice BLEU |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLORES-200 devtest | 1,012 | 52.70 | **55.37** | 24.12 | **28.28** |
| QnVoice conversation | 40 | 48.96 | **55.42** | 24.23 | **31.26** |
| OPUS Tatoeba v2026-07-08 | 424 | **55.02** | 41.00 | **27.74** | 14.52 |

The paired 95% intervals for the QnVoice chrF++ difference were +1.97 to +3.36 on FLORES-200, +1.35 to +11.85 on the conversation set, and −16.74 to −11.36 on Tatoeba. Both decoders returned text for every input. The short phrases in Tatoeba and manual review exposed mistranslations, especially where the wording was idiomatic. The conversation references were written for this experiment and have no independent native-speaker assessment. [Full scores, settings, and artifact hashes](benchmarks/results.json)

The comparison used greedy decoding for the published GGUF files and QnVoice's quantized Helsinki ONNX export on an Apple Silicon Mac. The study excluded exact normalized English matches to the test sets from the adapter prompts. Exposure to these benchmarks during upstream training is unknown. [Helsinki checkpoint](https://huggingface.co/Helsinki-NLP/opus-mt-en-ceb/tree/e94aee550a4d15c5b5da28a4be4fb2cb0dede50a), [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200), [OPUS Tatoeba test archive](https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/ceb-en.txt.zip)

## Run offline

Use Python 3.11 or later, CMake 3.20 or later, and a C++ compiler. The download script verifies both models against their published SHA-256 checksums. The decoder uses the runtime revision recorded in the benchmark. [Pinned runtime source](https://github.com/tc-mb/llama.cpp-omni/tree/e0eca9d8f57ac8f5a5ad6191a8007b93436c1e71), [Model checksums](SHA256SUMS.txt)

~~~sh
python3 fetch-model.py
git clone https://github.com/tc-mb/llama.cpp-omni.git .deps/llama.cpp-omni
git -C .deps/llama.cpp-omni checkout --detach e0eca9d8f57ac8f5a5ad6191a8007b93436c1e71
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR="$PWD/.deps/llama.cpp-omni"
cmake --build build --target native_benchmark --parallel 4

printf '1\tCould you write the address down for me?\n' > build/example.tsv
./build/native_benchmark models/qnvoice-cebuano-compact.gguf \
  models/qnvoice-cebuano-adapter.gguf build/example.tsv 1
~~~

The decoder reads one sentence per line as an ID followed by a tab and the English text. It writes the same ID and the Cebuano translation to standard output. After setup, decoding uses the local model files.

## Replay the conversation benchmark

Recompute the published conversation scores from the 40 reference translations and archived outputs. [References](benchmarks/conversation.tsv), [Helsinki outputs](benchmarks/conversation-helsinki.json), [QnVoice outputs](benchmarks/conversation-qnvoice.tsv)

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python tools/score.py
~~~

To generate a fresh candidate output, extract the ID and English columns and use a batch size of 24, as in the reported run.

~~~sh
cut -f 1,2 benchmarks/conversation.tsv > build/conversation-input.tsv
./build/native_benchmark models/qnvoice-cebuano-compact.gguf \
  models/qnvoice-cebuano-adapter.gguf build/conversation-input.tsv 24 > build/conversation-qnvoice.tsv
python tools/score.py --candidate build/conversation-qnvoice.tsv
~~~

## Reuse and cite

The compact model uses Apache 2.0 and the QnVoice adapter uses CC BY 4.0. Both permit commercial reuse under their terms. [Licenses and attribution](LICENSES.md)

The translator preprint describes the model, benchmark protocol, and archived translation examples. [Translator preprint on Zenodo](https://doi.org/10.5281/zenodo.24045948), [Citation file](CITATION.cff)

The companion speech paper covers the separate Cebuano synthesis pipeline. [Cebuano speech preprint](https://doi.org/10.5281/zenodo.24032369)
