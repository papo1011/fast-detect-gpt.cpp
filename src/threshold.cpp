#include "../include/threshold.h"

#include "../include/io.h"

#include <cmath>

ThresholdResult find_optimal_threshold(std::shared_ptr<arrow::Table> table,
                                       const std::string &           score_col,
                                       const std::string &           label_col,
                                       double                        beta) {
    ThresholdResult best_res = { 0.0, 0.0, 0.0, 0.0, 0.0, true };

    auto scores = extract_column_as<double>(table, score_col);
    auto labels = extract_column_as<int>(table, label_col);

    if (scores.size() != labels.size() || scores.empty()) {
        std::cerr << "Error: Column mismatch or empty data." << std::endl;
        return best_res;
    }

    size_t n = scores.size();

    struct DataPoint {
        double score;
        int    label;
    };

    std::vector<DataPoint> data(n);
    int                    total_ai = 0;

    for (size_t i = 0; i < n; ++i) {
        data[i] = { scores[i], labels[i] };
        if (labels[i] == 1) {
            total_ai++;
        }
    }

    std::sort(data.begin(), data.end(), [](const DataPoint & a, const DataPoint & b) { return a.score < b.score; });

    int tp = 0;
    int fp = 0;
    int fn = total_ai;

    const double beta_sq = beta * beta;

    for (size_t i = 0; i < n; ++i) {
        if (data[i].label == 1) {
            tp++;
            fn--;
        } else {
            fp++;
        }

        const double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        const double recall    = total_ai > 0 ? (double) tp / total_ai : 0.0;

        double numerator   = (1 + beta_sq) * (precision * recall);
        double denominator = (beta_sq * precision) + recall;

        const double f_score = (denominator > 0) ? (numerator / denominator) : 0.0;

        if (f_score > best_res.f_score) {
            best_res.f_score         = f_score;
            best_res.precision       = precision;
            best_res.recall          = recall;
            best_res.threshold       = data[i].score;
            best_res.is_lower_better = true;
        }
    }

    return best_res;
}
