#include <iostream>
#include "llama.h"

struct LlamaState {
    llama_model* model;
    const llama_vocab *vocab;
    llama_context* ctx;
};

bool setup_llama(LlamaState& llama, const std::string& model_path) {
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 200; // if possible load all layers to GPU

    // Load the model from disk to memory
    llama.model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!llama.model) return false;

    llama.vocab = llama_model_get_vocab(llama.model);

    auto cparams = llama_context_default_params();
    // TODO: add a cli flag to set these values
    cparams.n_ctx = 4096;
    cparams.n_batch = 4096;

    cparams.embeddings = true;

    llama.ctx = llama_init_from_model(llama.model, cparams);
    return (llama.ctx != nullptr);
}

int main(const int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "How to use: ./fast-detect-gpt <model_path> \"Input Text\"" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    std::string input_text = argv[2];

    llama_backend_init();

    LlamaState llama;
    if (!setup_llama(llama, model_path)) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return 1;
    }

    llama_backend_free();
    return 0;
}
