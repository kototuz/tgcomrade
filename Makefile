CXX=g++
CXXFLAGS=-Wall -Wextra -Wpedantic
LIBS_PATH=libs
export TG_API_ID
export TG_API_HASH
LLAMA_LIBS=-l:libggml-base.so    \
           -l:libggml-cpu.so     \
           -l:libggml.so         \
           -l:libllama.so        \
           -l:libllava_shared.so \
           -l:libmtmd_shared.so
TD_LIBS=-ltdclient       \
        -ltdapi          \
        -ltdcore         \
        -ltddb           \
        -ltdjson_private \
        -ltdjson_static  \
        -ltdmtproto      \
        -ltdnet          \
        -ltdsqlite       \
        -ltdutils        \
        -ltdactor
OTHER_LIBS=-lm -lz -lssl -lcrypto

all: tgcomrade txt2bpe

tgcomrade: build src/tgcomrade.cpp
	$(CXX) $(CXXFLAGS) -DTG_API_ID=$(TG_API_ID) -DTG_API_HASH="\"$(TG_API_HASH)\"" -o build/tgcomrade src/tgcomrade.cpp -Iinclude -fPIC -L$(LIBS_PATH) $(TD_LIBS) $(OTHER_LIBS) -Wl,-rpath,$(LIBS_PATH) $(LLAMA_LIBS)

txt2bpe: build src/txt2bpe.cpp
	$(CXX) $(CXXFLAGS) -o build/txt2bpe src/txt2bpe.cpp

build:
	mkdir -p build
