#include <argparse/argparse.hpp>
#include "../include/detect.h"
#include "../include/utils.h"

int main(const int argc, char * argv[]) {
    argparse::ArgumentParser program("fast-detect-gpt", "0.1.0");

    program.add_argument("-m", "--model").help("Path to the GGUF model file").required();

    program.add_argument("-f", "--file").help("Path to the input text file").required();

    program.add_argument("-c", "--ctx").help("Size of the prompt context").default_value(4096).scan<'i', int>();

    program.add_argument("-b", "--batch").help("Logical maximum batch size for inference").default_value(4096).scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception & err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const auto model_path   = program.get<std::string>("--model");
    const auto input_file   = program.get<std::string>("--file");
    const int         n_ctx   = program.get<int>("--ctx");
    const int         n_batch = program.get<int>("--batch");

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
    if (!setup_llama(llama, model_path, n_ctx, n_batch)) {
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

    if (n_tokens > n_ctx) {
        std::cerr << "Too many tokens provided: " << n_tokens << " (maximum " << n_ctx << ")" << std::endl;
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
