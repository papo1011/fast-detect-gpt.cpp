#include "../include/detect.h"

#include <cmath>

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