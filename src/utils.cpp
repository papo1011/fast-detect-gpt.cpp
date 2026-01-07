#include "../include/utils.h"
#include <fstream>
#include <sstream>

bool setup_llama(LlamaState & llama, const std::string & model_path, int n_ctx, int n_batch) {
    auto mparams         = llama_model_default_params();
    // TODO: enable GPU support
    mparams.n_gpu_layers = 0;  // use CPU only for the moment

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

bool read_file_to_string(const std::string & path, std::string & out) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

void custom_log(ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}