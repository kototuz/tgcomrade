# tgcomrade

## Description

**tgcomrade** (Telegram Comrade) is a project that combines
two things: [td](https://github.com/tdlib/td), [llama.cpp](https://github.com/ggml-org/llama.cpp).
It is a telegram client that can reply to messages instead of you.

## Build

First of all you need to install libraries from [td](https://github.com/tdlib/td)
and [llama.cpp](https://github.com/ggml-org/llama.cpp). Install libraries using
guides from these projects. Move install libraries to `libs/` (create it before).

Build:
``` console
make TG_API_ID=<your-api-id> TG_API_HASH=<your-api-hash>
```

## Usage

``` console
./tgcomrade <chat-id> <model.gguf>
```
