#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>

struct repack_progress_state {
    int last_percent = -5;
};

static bool repack_progress(float progress, void * user_data) {
    auto * state = static_cast<repack_progress_state *>(user_data);
    const int percent = std::min(100, std::max(0, (int) (progress * 100.0f)));
    if (percent >= state->last_percent + 5 || percent == 100) {
        fprintf(stderr, "\rpersistent repack: %3d%%", percent);
        fflush(stderr);
        state->last_percent = percent;
        if (percent == 100) {
            fputc('\n', stderr);
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_REPACK)) {
        return 1;
    }
    if (params.out_file.empty()) {
        fprintf(stderr, "error: --output is required\n");
        return 1;
    }
    if (!params.repack_cache.empty() || !params.repack_file.empty()) {
        fprintf(stderr, "error: --repack-cache and --repack-file are load options, not llama-repack options\n");
        return 1;
    }
    if (params.n_gpu_layers == -1) {
        params.n_gpu_layers = 0;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto model_params = common_model_params_to_llama(params);
    repack_progress_state progress_state;
    if (model_params.progress_callback == nullptr) {
        model_params.progress_callback = repack_progress;
        model_params.progress_callback_user_data = &progress_state;
    }
    auto repack_params = llama_model_repack_default_params();
    repack_params.force = params.repack_force;
    repack_params.delete_source = params.repack_delete_source;

    fprintf(stderr, "persistent repack:\n  source: %s\n  destination: %s\n", params.model.path.c_str(), params.out_file.c_str());
    repack_progress(0.0f, &progress_state);
    const int result = llama_model_repack_to_file(
            params.model.path.c_str(), params.out_file.c_str(), model_params, repack_params);
    llama_backend_free();
    return result == 0 ? 0 : 1;
}
