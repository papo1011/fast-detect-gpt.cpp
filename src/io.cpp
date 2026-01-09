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

std::shared_ptr<arrow::Table> load_parquet_table(const std::string & path) {
    arrow::MemoryPool * pool = arrow::default_memory_pool();

    auto result_open = arrow::io::ReadableFile::Open(path);
    if (!result_open.ok()) {
        std::cerr << "Error opening file: " << result_open.status().ToString() << std::endl;
        return nullptr;
    }
    std::shared_ptr<arrow::io::ReadableFile> infile = *result_open;

    auto open_file_result = parquet::arrow::OpenFile(infile, pool);
    if (!open_file_result.ok()) {
        std::cerr << "Error creating Parquet reader: " << open_file_result.status().ToString() << std::endl;
        return nullptr;
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(open_file_result.ValueOrDie());

    std::shared_ptr<arrow::Table> table;
    auto                          status = reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Error reading table: " << status.ToString() << std::endl;
        return nullptr;
    }

    return table;
}

bool save_parquet_with_scores(const std::string &           out_path,
                              std::shared_ptr<arrow::Table> table,
                              const std::vector<double> &   scores) {
    if (!table) {
        std::cerr << "Error: Input table is null." << std::endl;
        return false;
    }

    if (table->num_rows() != static_cast<int64_t>(scores.size())) {
        std::cerr << "Error: Row count mismatch! Table: " << table->num_rows() << ", Scores: " << scores.size()
                  << std::endl;
        return false;
    }

    arrow::DoubleBuilder builder;

    auto status = builder.AppendValues(scores);
    if (!status.ok()) {
        std::cerr << "Error building score array: " << status.ToString() << std::endl;
        return false;
    }

    std::shared_ptr<arrow::Array> score_array;
    status = builder.Finish(&score_array);
    if (!status.ok()) {
        std::cerr << "Error finishing score array: " << status.ToString() << std::endl;
        return false;
    }

    const auto field = arrow::field("discrepancy", arrow::float64());

    const auto chunked_array = std::make_shared<arrow::ChunkedArray>(score_array);

    auto result_table = table->AddColumn(table->num_columns(), field, chunked_array);

    if (!result_table.ok()) {
        std::cerr << "Error adding column to table: " << result_table.status().ToString() << std::endl;
        return false;
    }

    std::shared_ptr<arrow::Table> new_table     = *result_table;
    auto                          result_create = arrow::io::FileOutputStream::Open(out_path);
    if (!result_create.ok()) {
        std::cerr << "Error creating output file: " << result_create.status().ToString() << std::endl;
        return false;
    }
    const std::shared_ptr<arrow::io::FileOutputStream> outfile = *result_create;

    status = parquet::arrow::WriteTable(*new_table, arrow::default_memory_pool(), outfile, 64 * 1024);

    if (!status.ok()) {
        std::cerr << "Error writing parquet file: " << status.ToString() << std::endl;
        return false;
    }

    return true;
}
