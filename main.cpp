#include "llama.h"

#include <argparse/argparse.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

double compute_discrepancy(const std::vector<float *> &     all_logits,
                           const std::vector<llama_token> & tokens,
                           int                              vocab_size) {
    double sum_ll   = 0.0;
    double sum_mean = 0.0;
    double sum_var  = 0.0;

    std::vector<double> buffer(vocab_size);

    const size_t steps = tokens.size() - 1;

    for (size_t t = 0; t < steps; t++) {
        const int     token_id = tokens[t + 1];
        const float * logits   = all_logits[t];

        auto [log_likelihood, mean, variance] = compute_token_stats(vocab_size, token_id, logits, buffer);

        sum_ll += log_likelihood;
        sum_mean += mean;
        sum_var += variance;
    }

    if (sum_var <= 1e-9) {
        return 0.0;
    }

    return (sum_ll - sum_mean) / std::sqrt(sum_var);
}

struct LlamaState {
    llama_model *       model;
    const llama_vocab * vocab;
    llama_context *     ctx;
};

bool setup_llama(LlamaState & llama, const std::string & model_path, int n_ctx) {
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
    cparams.n_batch    = n_ctx;
    cparams.embeddings = true;

    llama.ctx = llama_init_from_model(llama.model, cparams);
    return (llama.ctx != nullptr);
}

static bool read_file_to_string(const std::string & path, std::string & out) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

int main(int argc, char * argv[]) {
    argparse::ArgumentParser program("fast-detect-gpt", "0.1.0");

    program.add_argument("-m", "--model").help("Path to the GGUF model file").required();

    program.add_argument("-f", "--file").help("Path to the input text file").required();

    program.add_argument("-c", "--ctx").help("Size of the prompt context").default_value(4096).scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception & err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::string model_path   = program.get<std::string>("--model");
    std::string input_file   = program.get<std::string>("--file");
    int         n_ctx_size   = program.get<int>("--ctx");

    if (!std::filesystem::exists(input_file) || !std::filesystem::is_regular_file(input_file)) {
        std::cerr << "Input must be an existing regular file: " << input_file << std::endl;
        return 1;
    }

    std::string input_text;
    if (!read_file_to_string(input_file, input_text)) {
        std::cerr << "Failed to read input file: " << input_file << std::endl;
        return 1;
    }

    llama_backend_init();

    LlamaState llama = {};
    if (!setup_llama(llama, model_path, n_ctx_size)) {
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

    if (n_tokens > n_ctx_size) {
        std::cerr << "Too many tokens provided: " << n_tokens << " (maximum " << n_ctx_size << ")" << std::endl;
        return 1;
    }

    auto batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.n_tokens     = n_tokens;
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

    const int    vocab_size  = llama_vocab_n_tokens(llama.vocab);
    const double discrepancy = compute_discrepancy(logits_ptrs, tokens, vocab_size);

    std::cout << " DISCREPANCY: " << std::fixed << std::setprecision(4) << discrepancy << std::endl;

    llama_batch_free(batch);
    llama_free(llama.ctx);
    llama_model_free(llama.model);
    llama_backend_free();

    return 0;
}
