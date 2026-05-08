// Vanilla autoregressive greedy generation — reference golden generator.
// PERSISTENT-PROCESS VERSION: loads target once, then loops over prompts read
// from stdin (delimited by "\n<<END_PROMPT>>\n"). Each prompt's stats and
// TOKENS line are emitted between "=== PROMPT i BEGIN ===" and
// "=== PROMPT i END ===" markers.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static const std::string PROMPT_DELIM = "\n<<END_PROMPT>>\n";

static std::vector<std::string> read_prompts_from_stdin() {
    std::string buf((std::istreambuf_iterator<char>(std::cin)),
                    std::istreambuf_iterator<char>());
    std::vector<std::string> out;
    size_t start = 0;
    while (start < buf.size()) {
        size_t pos = buf.find(PROMPT_DELIM, start);
        if (pos == std::string::npos) {
            std::string p = buf.substr(start);
            if (!p.empty()) out.push_back(p);
            break;
        }
        std::string p = buf.substr(start, pos - start);
        if (!p.empty()) out.push_back(p);
        start = pos + PROMPT_DELIM.size();
    }
    return out;
}

static llama_token greedy_argmax(const float * logits, int n_vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return (llama_token) best;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) return 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init_tgt = common_init_from_params(params);
    llama_model   * model = init_tgt->model();
    llama_context * ctx   = init_tgt->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int MAX_NEW = params.n_predict > 0 ? params.n_predict : 128;

    auto prompts = read_prompts_from_stdin();
    LOG_INF("read %zu prompts from stdin\n", prompts.size());
    if (prompts.empty()) {
        LOG_ERR("no prompts on stdin\n");
        return 1;
    }

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        // Clear KV for fresh prompt.
        llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);

        std::vector<llama_token> inp = common_tokenize(ctx, prompts[pi], true, true);
        const int n_prompt = (int) inp.size();
        if (n_prompt < 2) {
            fprintf(stdout, "\n=== PROMPT %zu BEGIN ===\n", pi);
            fprintf(stdout, "ERROR: prompt too short (n_prompt=%d)\n", n_prompt);
            fprintf(stdout, "=== PROMPT %zu END ===\n", pi);
            fflush(stdout);
            continue;
        }
        llama_decode(ctx, llama_batch_get_one(inp.data(), n_prompt - 1));
        int n_past = n_prompt - 1;
        llama_token id_last = inp.back();

        std::vector<llama_token> out;
        out.reserve(MAX_NEW);

        auto t0 = ggml_time_us();
        for (int step = 0; step < MAX_NEW; ++step) {
            llama_decode(ctx, llama_batch_get_one(&id_last, 1));
            n_past += 1;
            const float * logits = llama_get_logits_ith(ctx, -1);
            llama_token tok = greedy_argmax(logits, n_vocab);
            out.push_back(tok);
            if (llama_vocab_is_eog(vocab, tok)) break;
            id_last = tok;
        }
        auto t1 = ggml_time_us();
        const double dt = (t1 - t0) / 1e6;

        fprintf(stdout, "\n=== PROMPT %zu BEGIN ===\n", pi);
        fprintf(stdout, "decoded %d tokens in %.3fs = %.2f tok/s\n",
                (int)out.size(), dt, out.size() / dt);
        fprintf(stdout, "TOKENS:");
        for (auto t : out) fprintf(stdout, " %d", (int)t);
        fprintf(stdout, "\n=== PROMPT %zu END ===\n", pi);
        fflush(stdout);
    }

    llama_backend_free();
    return 0;
}
