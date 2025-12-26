```bash
███████  █████  ███████ ████████    ██████  ███████ ████████ ███████  ██████ ████████     ██████  ██████ ████████
██      ██   ██ ██         ██       ██   ██ ██         ██    ██      ██         ██       ██       ██  ██    ██
█████   ███████ ███████    ██       ██   ██ █████      ██    █████   ██         ██       ██   ███ ██████    ██
██      ██   ██      ██    ██       ██   ██ ██         ██    ██      ██         ██       ██    ██ ██        ██
██      ██   ██ ███████    ██       ██████  ███████    ██    ███████  ██████    ██        ██████  ██        ██
```

# What is fast-detect-gpt.cpp?

It is a 200 lines wrapper around llama.cpp to enable detection of AI generated text.

### How to compile

```bash
   cmake .
   cmake --build ./build --target fast-detect-gpt -j 6
```