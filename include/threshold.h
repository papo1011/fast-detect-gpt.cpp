#pragma once

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <parquet/arrow/reader.h>

#include <iostream>

struct ThresholdResult {
    double threshold;
    double f1;
    double precision;
    double recall;
    double accuracy;
    bool   is_lower_better;
};

ThresholdResult find_optimal_threshold(std::shared_ptr<arrow::Table> table,
                                       const std::string &           score_col,
                                       const std::string &           label_col);
