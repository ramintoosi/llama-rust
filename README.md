# Llama Inference in Rust

---

This project is a work in progress. It aims to provide a simple implementation 
of a language model interface using Rust and the `llama.cpp` library.

## Features

- **Chat Mode**: Interact with the model in a conversational manner.
- **Completion Mode**: Generate text completions based on a given prompt.
- **Customizable Parameters**: Override model parameters, set the number of GPU layers, and more.
- **CUDA Support**: Run the model on the GPU or CPU.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ramintoosi/llama-rust
    cd llama-rust
    ```

2. Build the project:
    ```sh
    cargo build
    ```

## Usage

```sh
Usage: rllama [OPTIONS] --n-gpu-layers <N_GPU_LAYERS> <COMMAND>

Commands:
  local     Use an already downloaded model
  hf-model  Download a model from huggingface (or use a cached version)
  help      Print this message or the help of the given subcommand(s)

Options:
  -m, --mode <MODE>                    The mode of the code: completion or chat [default: chat] [possible values: chat, completion]
      --max-token <MAX_TOKEN>          set the length of the prompt + output in tokens [default: 512]
  -o <KEY_VALUE_OVERRIDES>             override some parameters of the model
  -g, --n-gpu-layers <N_GPU_LAYERS>    how many layers to keep on the gpu - zero is cpu mode (default: 0)
  -s, --seed <SEED>                    set the seed for the RNG [default: 561371]
      --threads <THREADS>              number of threads to use during generation (default: use all available threads)
      --threads-batch <THREADS_BATCH>  number of threads to use during batch and prompt processing (default: use all available threads)
  -c, --ctx-size <CTX_SIZE>            size of the prompt context (default: loaded from the model)
  -v, --verbose                        show the token/s speed at the end of each turn
  -h, --help                           Print help


```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) for the language model backend.

