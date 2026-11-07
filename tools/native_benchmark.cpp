// Offline, greedy decoding for a frozen QnVoice Cebuano GGUF candidate.
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int context_per_sequence = 512;
constexpr int output_limit = 192;

struct Input {
    std::string id;
    std::string text;
};

std::string prompt_for(std::string source) {
    // Keep literal input from introducing special chat markers.
    size_t position = 0;
    while ((position = source.find("<|", position)) != std::string::npos) {
        source.replace(position, 2, "‹|");
        position += 4;
    }
    return "<|im_start|>user\n"
        "请将以下文本翻译为宿务语，直接输出翻译结果，不要进行任何解释。\n\n" +
        source + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

std::string compact(std::string text) {
    for (char &ch : text) {
        if (ch == '\t' || ch == '\n' || ch == '\r') ch = ' ';
    }
    const auto first = text.find_first_not_of(' ');
    return first == std::string::npos ? "" : text.substr(first, text.find_last_not_of(' ') - first + 1);
}

void add(llama_batch & batch, llama_token token, int32_t position, llama_seq_id sequence, bool logits) {
    const auto i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = position;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = sequence;
    batch.logits[i] = logits ? 1 : 0;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & prompt) {
    std::vector<llama_token> tokens(prompt.size() + 16);
    int size = llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokens.data(), int(tokens.size()), false, true);
    if (size <= 0 || size + output_limit > context_per_sequence) throw std::runtime_error("Input exceeds the token limit");
    tokens.resize(size);
    return tokens;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    char buffer[256];
    const int length = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, false);
    if (length < 0) throw std::runtime_error("Token exceeds the output buffer");
    return {buffer, static_cast<size_t>(length)};
}

struct Batch {
    explicit Batch(int tokens) : value(llama_batch_init(tokens, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
    llama_batch value;
};

std::vector<std::string> translate(
    llama_context * context, const llama_vocab * vocab, Batch & scratch,
    const std::vector<Input> & inputs, size_t first, size_t count
) {
    auto & batch = scratch.value;
    llama_memory_clear(llama_get_memory(context), true);
    batch.n_tokens = 0;
    std::vector<int32_t> next_position(count), last_row(count);
    std::vector<bool> completed(count, false);
    std::vector<std::string> output(count);
    for (size_t i = 0; i < count; i++) {
        auto tokens = tokenize(vocab, prompt_for(inputs[first + i].text));
        for (size_t j = 0; j < tokens.size(); j++) {
            add(batch, tokens[j], int32_t(j), llama_seq_id(i), j + 1 == tokens.size());
        }
        last_row[i] = batch.n_tokens - 1;
        next_position[i] = int32_t(tokens.size());
    }
    const auto vocab_size = llama_vocab_n_tokens(vocab);
    for (int step = 0; step < output_limit && batch.n_tokens; step++) {
        if (llama_decode(context, batch) != 0) throw std::runtime_error("Decode failed");
        batch.n_tokens = 0;
        for (size_t i = 0; i < count; i++) {
            if (completed[i]) continue;
            const auto * logits = llama_get_logits_ith(context, last_row[i]);
            if (!logits) throw std::runtime_error("No decoder logits");
            const auto token = llama_token(std::max_element(logits, logits + vocab_size) - logits);
            if (llama_vocab_is_eog(vocab, token)) {
                completed[i] = true;
                continue;
            }
            output[i] += piece(vocab, token);
            last_row[i] = batch.n_tokens;
            add(batch, token, next_position[i]++, llama_seq_id(i), true);
        }
    }
    for (size_t i = 0; i < count; i++) output[i] = completed[i] ? compact(output[i]) : "";
    return output;
}

int run(int argc, char **argv) {
    if (argc != 5) throw std::runtime_error("Usage: native_benchmark MODEL ADAPTER_OR_DASH INPUT_TSV BATCH_SIZE");
    const int count = std::stoi(argv[4]);
    if (count < 1 || count > 32) throw std::runtime_error("Batch size must be 1 to 32");
    std::ifstream file(argv[3]);
    if (!file) throw std::runtime_error("Could not open the input file");
    std::vector<Input> inputs;
    for (std::string line; std::getline(file, line);) {
        auto separator = line.find('\t');
        if (separator == std::string::npos) throw std::runtime_error("Expected TSV input");
        inputs.push_back({line.substr(0, separator), compact(line.substr(separator + 1))});
    }
    if (inputs.empty()) throw std::runtime_error("No input records");

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free
    );
    if (!model) throw std::runtime_error("Could not load the base model");
    // llama_model_free owns any adapter loaded for this model.
    auto adapter = std::string(argv[2]) == "-" ? nullptr : llama_adapter_lora_init(model.get(), argv[2]);
    if (!adapter && std::string(argv[2]) != "-") throw std::runtime_error("Could not load the adapter");
    auto cp = llama_context_default_params();
    cp.n_ctx = count * context_per_sequence;
    cp.n_seq_max = count;
    cp.n_batch = count * context_per_sequence;
    cp.n_ubatch = std::min(count * context_per_sequence, 1024);
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.no_perf = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
    if (!context) throw std::runtime_error("Could not create a context");
    if (adapter) {
        float scale = 1.0;
        if (llama_set_adapters_lora(context.get(), &adapter, 1, &scale) != 0) {
            throw std::runtime_error("Could not apply the adapter");
        }
    }
    auto vocab = llama_model_get_vocab(model.get());
    Batch scratch(cp.n_batch);
    const auto start = std::chrono::steady_clock::now();
    for (size_t offset = 0; offset < inputs.size(); offset += count) {
        auto result = translate(context.get(), vocab, scratch, inputs, offset, std::min(size_t(count), inputs.size() - offset));
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << inputs[offset + i].id << '\t' << result[i] << '\n';
        }
        std::cout.flush();
        if (offset == 0 || (offset / count) % 4 == 3) {
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cerr << offset + result.size() << '/' << inputs.size() << " in " << elapsed << " s\n";
        }
    }
    return 0;
}
}

int main(int argc, char **argv) {
    llama_log_set([](ggml_log_level level, const char * message, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) std::cerr << message;
    }, nullptr);
    llama_backend_init();
    int status = 0;
    try {
        status = run(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    llama_backend_free();
    return status;
}
