// Snippets from https://github.com/tsoding/bpe

#include <cstdio>
#include <clocale>
#include <cstdint>
#include <vector>
#include <cassert>
#include <fstream>
#include <unordered_map>
#include <cstring>

#define OUTPUT_FILE "out.bpe"

struct Token {
    uint32_t value; // may be either `symbol` or `pair_id`
    bool is_node;
};

struct Pair {
    Token l, r;
    uint64_t freq;
};

struct Pair_Key {
    Token l, r;

    bool operator==(const Pair_Key &other) const {
        return l.value == other.l.value &&
               r.value == other.r.value;
    }
};

template <>
struct std::hash<Pair_Key>
{
    std::size_t operator()(const Pair_Key& k) const
    {
        union {
            uint32_t lr[2];
            std::size_t hash;
        } u;

        u.lr[0] = k.l.value;
        u.lr[0] = k.r.value;

        return u.hash;
    }
};


bool parse_tokens_from_file(const char *path, std::vector<Token> &result)
{
    std::ifstream ifs(path);
    std::string bytes((std::istreambuf_iterator<char>(ifs)),
                        (std::istreambuf_iterator<char>()));

    size_t mbslen = mbstowcs(NULL, bytes.c_str(), 0);
    if (mbslen == (size_t)-1) {
        fprintf(stderr, "ERROR: Invalid multibyte sequence encountered\n");
        return false;
    }

    mbslen += 1;
    std::vector<wchar_t> wstr(mbslen);
    size_t new_len = mbstowcs(wstr.data(), bytes.c_str(), mbslen);
    assert(new_len != (size_t)-1);
    wstr.resize(new_len);

    result.reserve(new_len);
    for (size_t i = 0; i < new_len; i++) {
        result.push_back({(uint32_t)wstr[i], false});
    }

    return true;
}

std::unordered_map<Pair_Key, size_t> collect_freqs(std::vector<Token> &tokens_in)
{
    std::unordered_map<Pair_Key, size_t> result;
    for (size_t i = 0; i < tokens_in.size() - 1; i++) {
        Pair_Key pair = {tokens_in[i], tokens_in[i + 1]};
        if (result.count(pair)) result[pair] += 1;
        else result.insert({pair, 1});
    }

    return result;
}

void print_token(Token token)
{
    if (token.is_node) {
        printf("[%u]", token.value);
    } else {
        printf("%lc", token.value);
    }
}

void print_tokens(std::vector<Token> *tokens)
{
    for (size_t i = 0; i < tokens->size(); i++) {
        print_token(tokens->at(i));
    }
    printf("\n");
}

void print_freq(std::unordered_map<Pair_Key, size_t> &freq)
{
    for (auto const &it : freq) {
        printf("(");
        print_token(it.first.l);
        printf(",");
        print_token(it.first.r);
        printf(") => %zu\n", it.second);
    }
}

void pair_dec_freq_or_remove(Pair_Key k, std::unordered_map<Pair_Key, size_t> &freq)
{
    auto pair_freq = freq.find(k);
    assert(pair_freq != freq.end());
    assert(pair_freq->second > 0);
    pair_freq->second -= 1;
    if (pair_freq->second == 0) {
        freq.erase(k);
    }
}

bool write_entire_file(const char *path, const void *data, size_t size)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: Could not open file %s for writing: %s\n", path, strerror(errno));
        return false;
    }

    const char *buf = (const char *)data;
    while (size > 0) {
        size_t n = fwrite(buf, 1, size, f);
        if (ferror(f)) {
            fprintf(stderr, "ERROR: Could not write into file %s: %s\n", path, strerror(errno));
            fclose(f);
            return false;
        }
        size -= n;
        buf  += n;
    }

    return true;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input-file>\n", argv[0]);
        return 1;
    }

    setlocale(LC_ALL, "");

    std::vector<Token> tokens_in_buf;
    parse_tokens_from_file(argv[1], tokens_in_buf);
    std::vector<Token> *tokens_in = &tokens_in_buf;

    std::vector<Token> tokens_out_buf;
    std::vector<Token> *tokens_out = &tokens_out_buf;

    std::unordered_map<Pair_Key, size_t> freq = collect_freqs(tokens_in_buf);

    std::vector<Pair> pairs;
    while (true) {
        // print_tokens(tokens_in);
        // print_tokens(tokens_out);
        // print_freq(freq);
        // puts("==============================");
        // printf("INFO: Token count: %zu\n", tokens_in->size());
        // printf("INFO: Pair count:  %zu\n", pairs.size());

        Pair_Key max_pair;
        size_t max_freq = 0;
        for (const auto &it : freq) {
            if (it.second > max_freq) {
                max_freq = it.second;
                max_pair = it.first;
            }
        }

        // print_token(max_pair.l); puts("");
        // print_token(max_pair.r); puts("");

        if (max_freq <= 1) break; // compression is done

        uint32_t new_token_id = pairs.size();
        pairs.push_back({max_pair.l, max_pair.r, max_freq});

        tokens_out->clear();
        for (size_t i = 0; i < tokens_in->size();) {
            if (i + 1 >= tokens_in->size()) {
                tokens_out->push_back(tokens_in->at(i));
                i += 1;
            } else {
                Pair_Key pair;
                pair.l = tokens_in->at(i);
                pair.r = tokens_in->at(i + 1);
                if (memcmp(&pair, &max_pair, sizeof(pair)) == 0) {
                    print_token(pair.l); puts("");
                    print_token(pair.r); puts("");
                    puts("OK!");
                    std::unordered_map<Pair_Key, size_t>::iterator pair_freq;
                    if (tokens_out->size() > 0) {
                        pair.l = tokens_out->at(tokens_out->size() - 1);

                        pair.r = tokens_in->at(i);
                        pair_dec_freq_or_remove(pair, freq);

                        pair.r = {new_token_id, true};
                        if (!freq.count(pair)) freq.insert({pair, 1});
                        else freq[pair] += 1;
                    }

                    pair = max_pair;
                    pair_dec_freq_or_remove(pair, freq);

                    tokens_out->push_back({new_token_id, true});
                    i += 2;

                    //         v
                    // in:  abcd
                    // out: aZ
                    // Z=bc
                    if (i < tokens_in->size()) {
                        pair.r = tokens_in->at(i);

                        pair.l = tokens_in->at(i-1);
                        pair_dec_freq_or_remove(pair, freq);

                        pair.l = tokens_out->at(tokens_out->size()-1);
                        if (!freq.count(pair)) freq.insert({pair, 1});
                        else freq[pair] += 1;
                    }
                } else {
                    tokens_out->push_back(tokens_in->at(i));
                    i += 1;
                }
            }
        }

        auto *tmp = tokens_in;
        tokens_in = tokens_out;
        tokens_out = tmp;
    }

    // print_freq(freq);
    //
    // for (size_t i = 0; i < tokens_in->size(); i++) {
    //     Token token = tokens_in->at(i);
    //     print_token(token);
    // }

    printf("INFO: Generated %zu pairs\n", pairs.size());

    if (!write_entire_file(OUTPUT_FILE, pairs.data(), pairs.size()*sizeof(Pair)))
        return 1;

    return 0;
}
