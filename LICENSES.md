# Licenses and attribution

The QnVoice scripts in this repository use Apache License 2.0. The QnVoice conversation benchmark examples and accompanying QnVoice research documentation use CC BY 4.0. Both licenses allow commercial reuse. [Apache License 2.0](LICENSE), [CC BY 4.0](LICENSE-CC-BY-4.0.txt)

The two published GGUF files have separate terms.

| Artifact | License | Attribution |
| --- | --- | --- |
| qnvoice-cebuano-compact.gguf | Apache 2.0 | The Index LLM Team developed Index-Translate-2B and published the Q4_K_M conversion. QnVoice reduced its vocabulary and remapped the retained token IDs. [Index-Translate-2B](https://huggingface.co/IndexTeam/Index-Translate-2B/tree/9bc08445315723eb9b77d1b93505915eb03c3f01), [upstream GGUF](https://huggingface.co/IndexTeam/Index-Translate-2B-GGUF/tree/3a0eca6ce931d88d00614d427ab3ed8ff0743902) |
| qnvoice-cebuano-adapter.gguf | CC BY 4.0 | © 2026 QnVoice, QnSub Software Solutions. When redistributing or adapting the adapter, credit QnVoice, link to the license, and indicate changes. [Model license notice](https://huggingface.co/CholoAsis/qnvoice-cebuano-translation/blob/27efd4da6be61a51219234ca584586cb084a6133/LICENSES.md) |

Adapter training used original QnVoice examples and synthetic Cebuano targets generated locally with the Apache 2.0 Index MLX conversion. Some English prompts came from the Tatoeba CC0 export. The research used FLORES-200 and OPUS Tatoeba as held-out benchmarks. The archived outputs in this repository come from the original QnVoice conversation examples. [Index MLX conversion](https://huggingface.co/yibie/Index-Translate-2B-MLX-4bit/tree/47de2e96d051cc52ad9459cd78cc578ba6311d41), [Tatoeba downloads](https://tatoeba.org/en/downloads), [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200)

Suggested credit is QnVoice Offline English-to-Cebuano Translation, QnSub Software Solutions (2026). The Cebuano speech paper describes the work that preceded this translation release. [Citation metadata](CITATION.cff), [Zenodo paper](https://doi.org/10.5281/zenodo.24032369)
