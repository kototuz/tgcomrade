#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string.h>
#include <ctype.h>
#include <vector>

#include <llama.h>

#include <td/telegram/Client.h>
namespace td_api = td::td_api;

#define TG_WAIT_TIME 10.0

#define LLAMA_GPU_LAYER_COUNT 99
#define LLAMA_CONTEXT_SIZE    2048

#define LIST_OF_UPDATE_HANDLERS \
    X(updateAuthorizationState, update_auth_state) \
    X(authorizationStateWaitTdlibParameters, auth_state_wait_tdlib_params) \
    X(authorizationStateReady, auth_state_ready) \
    X(authorizationStateWaitPhoneNumber, auth_state_wait_phone_number) \
    X(authorizationStateWaitCode, auth_state_wait_code) \
    X(updateNewMessage, update_new_message) \

static void auth_state_wait_code(td_api::object_ptr<td_api::authorizationStateWaitCode>);
static void auth_state_wait_phone_number(td_api::object_ptr<td_api::authorizationStateWaitPhoneNumber>);
static void auth_state_ready(td_api::object_ptr<td_api::authorizationStateReady>);
static void auth_state_wait_tdlib_params(td_api::object_ptr<td_api::authorizationStateWaitTdlibParameters>);
static void update_auth_state(td_api::object_ptr<td_api::updateAuthorizationState>);
static void update_new_message(td_api::object_ptr<td_api::updateNewMessage>);

static void process_update(td_api::object_ptr<td_api::Object> u);
static bool str_to_int64(const char *str, size_t len, std::int64_t *res);
static bool load_model(const char *model);
static bool generate_response(const std::string &prompt, std::string &res);

static td::ClientManager manager;
static std::int32_t      client_id;
static std::int64_t      chat_id;
static std::int64_t      user_id;

static llama_model *model;
static const llama_vocab *vocab;
static llama_context *ctx;
static llama_sampler *smpl;
static std::vector<llama_chat_message> messages;
static std::vector<char> formatted;
static int prev_formatted_len = 0;

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <chat-id> <model.gguf>\n", argv[0]);
        return 1;
    }

    if (!str_to_int64(argv[1], strlen(argv[1]), &chat_id)) {
        fputs("ERROR: Invalid chat id integer\n", stderr);
        return 1;
    }

    if (!load_model(argv[2])) return 1;

    messages.push_back({"system", "Answer in a conversational style"});

    // Initialize client
    td::ClientManager::execute(td_api::make_object<td_api::setLogVerbosityLevel>(1));
    client_id = manager.create_client_id();
    manager.send(client_id, 1, td_api::make_object<td_api::getOption>("version"));

    // Receive events
    while (true) {
        auto resp = manager.receive(TG_WAIT_TIME);
        if (resp.object == nullptr) continue;

        if (resp.request_id == 0) {
            process_update(std::move(resp.object));
        } else {
            switch (resp.object->get_id()) {
            case td_api::error::ID:
                std::cout << "ERROR: " <<
                    static_cast<td_api::error&>(*resp.object).message_ << "\n";
                break;

            case td_api::user::ID:
                user_id = static_cast<td_api::user&>(*resp.object).id_;
                break;
            }
        }
    }

    return 0;
}


static bool generate_response(const std::string &input, std::string &res)
{
    const char *tmpl = llama_model_chat_template(model, nullptr);

    messages.push_back({"user", strdup(input.c_str())});
    int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (new_len > (int)formatted.size()) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    }
    if (new_len < 0) {
        fputs("ERROR: Could not apply chat template\n", stderr);
        return false;
    }

    std::string prompt(formatted.begin() + prev_formatted_len, formatted.begin() + new_len);
    
    const bool is_first = llama_kv_self_used_cells(ctx) == 0;

    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        fputs("ERROR: Could not tokenize the prompt\n", stderr);
        return false;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    printf(">> ");
    while (true) {
        int n_ctx = llama_n_ctx(ctx);
        int n_ctx_used = llama_kv_self_used_cells(ctx);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            fputs("ERROR: Context size exceeded\n", stderr);
            return false;
        }

        if (llama_decode(ctx, batch)) {
            fputs("ERROR: Could not decode\n", stderr);
            return false;
        }

        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fputs("ERROR: Could not convert token to piece\n", stderr);
            return false;
        }

        printf("%.*s", n, buf);
        fflush(stdout);

        res.append(buf, n);

        batch = llama_batch_get_one(&new_token_id, 1);
    }

    putchar('\n');

    messages.push_back({"assistant", strdup(res.c_str())});
    prev_formatted_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
    if (prev_formatted_len < 0) {
        fputs("ERROR: Could not apply chat template\n", stderr);
        return false;
    }

    return true;
}

static bool load_model(const char *model_path)
{
    puts("Loading model...");

    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = LLAMA_GPU_LAYER_COUNT;

    model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fputs("ERROR: Could not load model from file\n", stderr);
        return false;
    }

    vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = LLAMA_CONTEXT_SIZE;
    ctx_params.n_batch = LLAMA_CONTEXT_SIZE;

    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fputs("ERROR: Could not create context\n", stderr);
        return false;
    }

    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    formatted = std::vector<char>(llama_n_ctx(ctx));

    return true;
}

static void process_update(td_api::object_ptr<td_api::Object> u)
{
    switch (u->get_id()) {
#define X(update_type, handler) \
        case td_api::update_type::ID: \
            handler(td_api::move_object_as<td_api::update_type>(u)); \
            break;
        LIST_OF_UPDATE_HANDLERS
#undef X
    }
}

static bool str_to_int64(const char *str, size_t len, std::int64_t *res)
{
    for (size_t i = 0; i < len; i++) {
        if (!isdigit(str[i])) return false;
        *res = *res*10 + (str[i]-'0');
    }

    return true;
}

static void update_new_message(td_api::object_ptr<td_api::updateNewMessage> u)
{
    if (u->message_->chat_id_ != chat_id) return;
    if (u->message_->sender_id_->get_id() == td_api::messageSenderUser::ID) {
        if (static_cast<td_api::messageSenderUser&>(*u->message_->sender_id_).user_id_ == user_id)
            return;
    }

    if (u->message_->content_->get_id() == td_api::messageText::ID) {
        auto send_message = td_api::make_object<td_api::sendMessage>();
        send_message->chat_id_ = chat_id;
        auto message_content = td_api::make_object<td_api::inputMessageText>();
        send_message->reply_to_ =
            td_api::make_object<td_api::inputMessageReplyToMessage>(
                    u->message_->id_, nullptr);

        // manager.send(client_id, 1,
        //         td_api::make_object<td_api::sendChatAction>(chat_id, 0, nullptr,
        //             td_api::make_object<td_api::chatActionTyping>()));

        std::string resp;
        message_content->text_ = td_api::make_object<td_api::formattedText>();
        if (generate_response(static_cast<td_api::messageText &>(*u->message_->content_).text_->text_, resp)) {
            message_content->text_->text_ = std::move(resp);
        } else {
            message_content->text_->text_ = "Sorry, something went wrong";
        }

        // manager.send(client_id, 1,
        //         td_api::make_object<td_api::sendChatAction>(chat_id, 0, nullptr,
        //             td_api::make_object<td_api::chatActionCancel>()));

        send_message->input_message_content_ = std::move(message_content);
        manager.send(client_id, 1, std::move(send_message));
    }
}

static void update_auth_state(td_api::object_ptr<td_api::updateAuthorizationState> u)
{
    process_update(std::move(u->authorization_state_));
}

static void auth_state_wait_tdlib_params(td_api::object_ptr<td_api::authorizationStateWaitTdlibParameters>)
{
    auto params = td_api::make_object<td_api::setTdlibParameters>();
    params->use_test_dc_ = false;
    params->database_directory_ = "data";
    params->use_file_database_ = true;
    params->use_chat_info_database_ = true;
    params->use_message_database_ = true;
    params->use_secret_chats_ = false;
    params->api_id_ = TG_API_ID;
    params->api_hash_ = TG_API_HASH;
    params->system_language_code_ = "en";
    params->device_model_ = "Desktop";
    params->system_version_ = "Debian 12";
    params->application_version_ = "0.1";
    puts("Sending tdlib parameters...");
    manager.send(client_id, 1, td_api::move_object_as<td_api::Function>(params));
}

static void auth_state_wait_phone_number(td_api::object_ptr<td_api::authorizationStateWaitPhoneNumber>)
{
    std::string input;
    printf("Phone number: ");
    std::getline(std::cin, input);
    puts("Sending phone number...");
    manager.send(client_id, 1, td_api::make_object<td_api::setAuthenticationPhoneNumber>(input, nullptr));
}

static void auth_state_wait_code(td_api::object_ptr<td_api::authorizationStateWaitCode>)
{
    std::string input;
    printf("Code: ");
    std::getline(std::cin, input);
    puts("Sending code...");
    manager.send(client_id, 1, td_api::make_object<td_api::checkAuthenticationCode>(input));
}

static void auth_state_ready(td_api::object_ptr<td_api::authorizationStateReady>)
{
    puts("Succesful login");
    manager.send(client_id, 1, td_api::make_object<td_api::getMe>());
}
