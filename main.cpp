#include <iostream>
#include "llama.h"

int main(const int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "How to use: ./fast-detect-gpt <model_path> \"Input Text\"" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_text = argv[2];

    llama_backend_init();

    auto project = "Fast Detect GPT";
    std::cout << "Hello and welcome to " << project << "!\n";

    llama_backend_free();
    return 0;
}
