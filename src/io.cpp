#include "../include/io.h"

#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <fstream>
#include <iostream>
#include <sstream>

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

std::pair<std::shared_ptr<arrow::Table>, std::vector<std::string>> load_parquet_and_get_text(
    const std::string & path,
    const std::string & col_name) {
    std::vector<std::string> texts;
    arrow::MemoryPool *      pool = arrow::default_memory_pool();

    auto result_open = arrow::io::ReadableFile::Open(path);
    if (!result_open.ok()) {
        std::cerr << "Error opening file: " << result_open.status().ToString() << std::endl;
        return { nullptr, texts };
    }
    const std::shared_ptr<arrow::io::ReadableFile> infile = *result_open;

    auto open_file_result = parquet::arrow::OpenFile(infile, pool);

    if (!open_file_result.ok()) {
        std::cerr << "Error creating Parquet reader: " << open_file_result.status().ToString() << std::endl;
        return { nullptr, texts };
    }

    const std::unique_ptr<parquet::arrow::FileReader> reader = std::move(open_file_result.ValueOrDie());
    std::shared_ptr<arrow::Table>                     table;
    if (const auto status = reader->ReadTable(&table); !status.ok()) {
        std::cerr << "Error reading table: " << status.ToString() << std::endl;
        return { nullptr, texts };
    }

    const auto col = table->GetColumnByName(col_name);
    if (!col) {
        std::cerr << "Column '" << col_name << "' not found in Parquet!" << std::endl;
        return { table, texts };
    }

    for (int i = 0; i < col->num_chunks(); i++) {
        const auto array = std::static_pointer_cast<arrow::StringArray>(col->chunk(i));
        for (int64_t j = 0; j < array->length(); j++) {
            if (array->IsNull(j)) {
                texts.push_back("");
            } else {
                texts.push_back(array->GetString(j));
            }
        }
    }

    return { table, texts };
}
