#pragma once
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <parquet/arrow/reader.h>

bool read_file_to_string(const std::string & path, std::string & out);

std::pair<std::shared_ptr<arrow::Table>, std::vector<std::string>> load_parquet_and_get_text(
    const std::string & path,
    const std::string & col_name);
