#include "llama.h"

#include <math.h>

#include <iostream>
#include <vector>

struct TokenStats {
    double log_likelihood;
    double mean;
    double variance;
};

TokenStats compute_token_stats(const int             vocab_size,
                               const int             token_id,
                               const float *         logits,
                               std::vector<double> & buffer) {
    float max_logit = -1e9;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        buffer[i] = std::exp(logits[i] - max_logit);
        sum_exp += buffer[i];
    }
    double log_sum_exp = std::log(sum_exp);

    TokenStats stats = { 0.0, 0.0, 0.0 };

    if (token_id >= 0 && token_id < vocab_size) {
        stats.log_likelihood = (logits[token_id] - max_logit) - log_sum_exp;
    }

    double mean            = 0.0;  // E[X]
    double expected_square = 0.0;  // E[X^2]

    for (int i = 0; i < vocab_size; i++) {
        double p  = buffer[i] / sum_exp;
        double lp = (logits[i] - max_logit) - log_sum_exp;

        mean += p * lp;
        expected_square += p * (lp * lp);
    }

    stats.mean     = mean;
    stats.variance = expected_square - (mean * mean);  // E[X^2] - (E[X])^2

    return stats;
}

struct LlamaState {
    llama_model *       model;
    const llama_vocab * vocab;
    llama_context *     ctx;
};

bool setup_llama(LlamaState & llama, const std::string & model_path) {
    auto mparams         = llama_model_default_params();
    mparams.n_gpu_layers = 200;  // if possible load all layers to GPU

    // Load the model from disk to memory
    llama.model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!llama.model) {
        return false;
    }

    llama.vocab = llama_model_get_vocab(llama.model);

    auto cparams    = llama_context_default_params();
    // TODO: add a cli flag to set these values
    cparams.n_ctx   = 4096;
    cparams.n_batch = 4096;

    cparams.embeddings = true;

    llama.ctx = llama_init_from_model(llama.model, cparams);
    return (llama.ctx != nullptr);
}

int main(const int argc, char * argv[]) {
    if (argc < 3) {
        std::cerr << "How to use: ./fast-detect-gpt <model_path> \"Input Text\"" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    std::string       input_text = argv[2];

    llama_backend_init();

    LlamaState llama;
    if (!setup_llama(llama, model_path)) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return 1;
    }

    // Input text tokenized
    // size: input text length + 2 for BOS and EOS
    std::vector<llama_token> tokens(input_text.length() + 2);

    auto n_tokens =
        llama_tokenize(llama.vocab, input_text.c_str(), input_text.length(), tokens.data(), tokens.size(), true, false);

    if (n_tokens < 0) {
        n_tokens = -n_tokens;
        tokens.resize(n_tokens);
        n_tokens = llama_tokenize(llama.vocab, input_text.c_str(), input_text.length(), tokens.data(), tokens.size(),
                                  true, false);
    }
    tokens.resize(n_tokens);

    if (n_tokens < 2) {
        std::cerr << "Not enough tokens provided (minimum 2 tokens)" << std::endl;
        return 1;
    }

    // TODO: add cli flag to set ctx size
    if (n_tokens > 4096) {
        std::cerr << "Too many tokens provided (maximum 4096 tokens)" << std::endl;
        return 1;
    }

    auto batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true;
    }

    if (llama_decode(llama.ctx, batch) != 0) {
        std::cerr << "Inference failed" << std::endl;
        llama_batch_free(batch);
        return 1;
    }

    std::vector<float *> logits_ptrs;
    logits_ptrs.reserve(n_tokens);  // reserve space avoiding reallocations

    for (int i = 0; i < n_tokens; i++) {
        logits_ptrs.push_back(llama_get_logits_ith(llama.ctx, i));
    }

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
