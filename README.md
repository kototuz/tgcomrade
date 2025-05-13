# tgcomrade

## Dependencies

- [td](https://github.com/tdlib/td)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

## Description

**tgcomrade** (Telegram Comrade) is a program that generates response to
the specifific telegram chat for you.

Supported generators:
- gguf models
- bpe (see [bpe](https://github.com/tsoding/bpe))
    You can generate `bpe` files using `./build/txt2bpe`

## Build

First of all you need to install libraries from [td](https://github.com/tdlib/td)
and [llama.cpp](https://github.com/ggml-org/llama.cpp). Install libraries using
guides from these projects. Move install libraries to `libs/` (create it before).

Build:
``` console
make TG_API_ID=<your-api-id> TG_API_HASH=<your-api-hash>
```
