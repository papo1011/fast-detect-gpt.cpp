#include "../include/utils.h"

#include <iostream>

bool setup_llama(LlamaState & llama, const std::string & model_path, bool gpu, int n_ctx, int n_batch) {
    auto mparams = llama_model_default_params();

    if (gpu) {
        mparams.n_gpu_layers = 200;  // enable GPU
    } else {
        mparams.n_gpu_layers = 0;
    }

    llama.model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!llama.model) {
        return false;
    }

    llama.vocab = llama_model_get_vocab(llama.model);

    auto cparams       = llama_context_default_params();
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_batch;
    cparams.embeddings = true;

    llama.ctx = llama_init_from_model(llama.model, cparams);
    return (llama.ctx != nullptr);
}

void custom_log(ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

void print_logo() {
    const std::string logo = R"(
██████ ████  █████ ██████    █████  █████ ██████ █████ ████ ██████    █████   ██████ ██████
██    ██  ██ ██      ██      ██  ██ ██      ██   ██   ██      ██     ██       ██  ██   ██
████  ██████ █████   ██      ██  ██ ████    ██   ████ ██      ██     ██   ███ ██████   ██
██    ██  ██    ██   ██      ██  ██ ██      ██   ██   ██      ██     ██    ██ ██       ██
██    ██  ██ █████   ██      █████  █████   ██   █████ ████   ██      █████   ██       ██
    )";
    std::cout << "\033[36m" << logo << "\033[0m" << std::endl;
}

void signal_handler(const int signum) {
    std::cout << "\n\nInterrupt signal (" << signum << ") received." << std::endl;
    std::cout << "Stop inference loop and saving progress!" << std::endl;
    g_interrupted = true;
}
