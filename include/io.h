#pragma once
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <parquet/arrow/reader.h>

bool read_file_to_string(const std::string & path, std::string & out);

std::pair<std::shared_ptr<arrow::Table>, std::vector<std::string>> load_parquet_and_get_text(
    const std::string & path,
    const std::string & col_name);

std::shared_ptr<arrow::Table> load_parquet_table(const std::string & path);

bool save_parquet_with_scores(const std::string &           out_path,
                              std::shared_ptr<arrow::Table> table,
                              const std::vector<double> &   scores);

template <typename T> std::vector<T> extract_column_as(std::shared_ptr<arrow::Table> table, const std::string & name) {
    std::vector<T> values;
    if (!table) {
        return values;
    }

    auto col = table->GetColumnByName(name);
    if (!col) {
        return values;
    }

    for (int i = 0; i < col->num_chunks(); i++) {
        auto array = col->chunk(i);

        if (array->type_id() == arrow::Type::DOUBLE) {
            auto data = std::static_pointer_cast<arrow::DoubleArray>(array);
            for (int64_t j = 0; j < data->length(); j++) {
                values.push_back(static_cast<T>(data->Value(j)));
            }
        } else if (array->type_id() == arrow::Type::FLOAT) {
            auto data = std::static_pointer_cast<arrow::FloatArray>(array);
            for (int64_t j = 0; j < data->length(); j++) {
                values.push_back(static_cast<T>(data->Value(j)));
            }
        } else if (array->type_id() == arrow::Type::INT64) {
            auto data = std::static_pointer_cast<arrow::Int64Array>(array);
            for (int64_t j = 0; j < data->length(); j++) {
                values.push_back(static_cast<T>(data->Value(j)));
            }
        } else if (array->type_id() == arrow::Type::INT32) {
            auto data = std::static_pointer_cast<arrow::Int32Array>(array);
            for (int64_t j = 0; j < data->length(); j++) {
                values.push_back(static_cast<T>(data->Value(j)));
            }
        }
    }
    return values;
}
