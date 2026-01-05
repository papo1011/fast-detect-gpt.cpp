#pragma once
#include "llama.h"
#include <string>
#include <filesystem>

struct LlamaState {
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_context* ctx = nullptr;
};

bool read_file_to_string(const std::string& path, std::string& out);

bool setup_llama(LlamaState& llama, const std::string& model_path, int n_ctx, int n_batch);