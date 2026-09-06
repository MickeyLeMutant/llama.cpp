#include "llama.h"
#include "common.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct temporary_directory {
    fs::path path;

    ~temporary_directory() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

static std::vector<float> model_logits(const fs::path & path, bool use_extra_bufts = true) {
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_extra_bufts = use_extra_bufts;
    model_params.check_tensors = true;
    llama_model_ptr model(llama_model_load_from_file(path.string().c_str(), model_params));
    if (!model) {
        throw std::runtime_error("failed to load " + path.string());
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 8;
    llama_context_ptr context(llama_init_from_model(model.get(), context_params));
    if (!context) {
        throw std::runtime_error("failed to create context for " + path.string());
    }

    std::vector<llama_token> tokens = common_tokenize(llama_model_get_vocab(model.get()), "Hello", true, true);
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(context.get(), batch) != 0) {
        throw std::runtime_error("failed to decode with " + path.string());
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const float * logits = llama_get_logits_ith(context.get(), -1);
    if (logits == nullptr) {
        throw std::runtime_error("model returned no logits for " + path.string());
    }
    return std::vector<float>(logits, logits + n_vocab);
}

static void flip_byte(const fs::path & path, std::streamoff offset) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    char value;
    file.seekg(offset);
    file.get(value);
    if (!file) {
        throw std::runtime_error("failed to read " + path.string());
    }
    value ^= 1;
    file.seekp(offset);
    file.put(value);
    if (!file) {
        throw std::runtime_error("failed to modify " + path.string());
    }
}

static int test_persistent_repack(const fs::path & source_path) {
    temporary_directory temp {
        fs::temp_directory_path() / ("llama-repack-test-" +
                std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()))
    };
    fs::create_directories(temp.path);

    const fs::path repack_path = temp.path / "model.repack";
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto repack_params = llama_model_repack_default_params();
    if (llama_model_repack_to_file(source_path.string().c_str(), repack_path.string().c_str(), model_params, repack_params) != 0) {
        throw std::runtime_error("failed to create persistent repack");
    }
    if (llama_model_repack_validate_file(repack_path.string().c_str(), source_path.string().c_str(), model_params, true) != 0) {
        throw std::runtime_error("failed to validate persistent repack");
    }

    const std::vector<float> source_logits = model_logits(source_path);
    const std::vector<float> repack_logits = model_logits(repack_path);
    if (source_logits.size() != repack_logits.size() ||
            memcmp(source_logits.data(), repack_logits.data(), source_logits.size() * sizeof(float)) != 0) {
        throw std::runtime_error("persistent repack logits differ from runtime repack logits");
    }
    model_logits(source_path, false);
    bool placement_rejected = false;
    try {
        model_logits(repack_path, false);
    } catch (const std::exception &) {
        placement_rejected = true;
    }
    if (!placement_rejected) {
        throw std::runtime_error("persistent repack accepted an incompatible placement profile");
    }

    const fs::path corrupt_path = temp.path / "corrupt.repack";
    fs::copy_file(repack_path, corrupt_path);
    flip_byte(corrupt_path, 128);
    if (llama_model_ptr(llama_model_load_from_file(corrupt_path.string().c_str(), model_params))) {
        throw std::runtime_error("corrupt persistent repack loaded successfully");
    }

    const fs::path incompatible_path = temp.path / "incompatible.repack";
    fs::copy_file(repack_path, incompatible_path);
    flip_byte(incompatible_path, 12);
    if (llama_model_ptr(llama_model_load_from_file(incompatible_path.string().c_str(), model_params))) {
        throw std::runtime_error("incompatible persistent repack loaded successfully");
    }

    const fs::path truncated_path = temp.path / "truncated.repack";
    fs::copy_file(repack_path, truncated_path);
    fs::resize_file(truncated_path, fs::file_size(truncated_path) / 2);
    if (llama_model_ptr(llama_model_load_from_file(truncated_path.string().c_str(), model_params))) {
        throw std::runtime_error("truncated persistent repack loaded successfully");
    }

    const fs::path data_corrupt_path = temp.path / "data-corrupt.repack";
    fs::copy_file(repack_path, data_corrupt_path);
    flip_byte(data_corrupt_path, (std::streamoff) fs::file_size(data_corrupt_path) - 1);
    if (llama_model_repack_validate_file(data_corrupt_path.string().c_str(), nullptr, model_params, true) == 0) {
        throw std::runtime_error("persistent repack data corruption was not detected");
    }

    const fs::path changed_source_path = temp.path / "changed-source.gguf";
    fs::copy_file(source_path, changed_source_path);
    flip_byte(changed_source_path, (std::streamoff) fs::file_size(changed_source_path) - 1);
    if (llama_model_repack_validate_file(repack_path.string().c_str(), changed_source_path.string().c_str(), model_params, false) == 0) {
        throw std::runtime_error("persistent repack accepted a different source fingerprint");
    }

    const fs::path delete_source_path = temp.path / "delete-source.gguf";
    const fs::path delete_repack_path = temp.path / "delete-source.repack";
    fs::copy_file(source_path, delete_source_path);
    repack_params.delete_source = true;
    if (llama_model_repack_to_file(delete_source_path.string().c_str(), delete_repack_path.string().c_str(), model_params, repack_params) != 0 ||
            fs::exists(delete_source_path) || !fs::exists(delete_repack_path)) {
        throw std::runtime_error("safe source deletion failed");
    }
    model_logits(delete_repack_path);

    const fs::path retained_source_path = temp.path / "retained-source.gguf";
    const fs::path existing_path = temp.path / "existing.repack";
    fs::copy_file(source_path, retained_source_path);
    std::ofstream(existing_path, std::ios::binary).put('x');
    if (llama_model_repack_to_file(retained_source_path.string().c_str(), existing_path.string().c_str(), model_params, repack_params) == 0 ||
            !fs::exists(retained_source_path)) {
        throw std::runtime_error("failed conversion removed its source");
    }

    const fs::path cache_path = temp.path / "cache.repack";
    common_params cache_params;
    cache_params.model.path = source_path.string();
    cache_params.repack_cache = cache_path.string();
    cache_params.fit_params = false;
    cache_params.n_gpu_layers = 0;
    auto first_load = common_init_from_params(cache_params, true);
    if (!first_load || first_load->model() == nullptr || !fs::exists(cache_path)) {
        throw std::runtime_error("automatic persistent repack cache creation failed");
    }
    first_load.reset();
    const auto old_time = fs::last_write_time(cache_path) - std::chrono::hours(24);
    fs::last_write_time(cache_path, old_time);
    const auto cache_marker = fs::last_write_time(cache_path);
    auto second_load = common_init_from_params(cache_params, true);
    if (!second_load || second_load->model() == nullptr || fs::last_write_time(cache_path) != cache_marker) {
        throw std::runtime_error("compatible persistent repack cache was not reused");
    }
    second_load.reset();
    flip_byte(cache_path, 128);
    auto rebuilt_load = common_init_from_params(cache_params, true);
    if (!rebuilt_load || rebuilt_load->model() == nullptr ||
            llama_model_repack_validate_file(cache_path.string().c_str(), source_path.string().c_str(), model_params, true) != 0) {
        throw std::runtime_error("incompatible persistent repack cache was not rebuilt");
    }

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[] ) {
    if (argc == 3 && strcmp(argv[1], "--persistent-repack") == 0) {
        llama_backend_init();
        try {
            const int result = test_persistent_repack(argv[2]);
            llama_backend_free();
            return result;
        } catch (const std::exception & exception) {
            fprintf(stderr, "persistent repack test failed: %s\n", exception.what());
            llama_backend_free();
            return EXIT_FAILURE;
        }
    }

    auto * model_path = common_get_model_or_exit(argc, argv);
    auto * file = fopen(model_path, "r");
    if (file == nullptr) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "using '%s'\n", model_path);
    fclose(file);

    llama_backend_init();
    auto params = llama_model_params{};
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.progress_callback = [](float progress, void * ctx){
        (void) ctx;
        return progress > 0.50;
    };
    auto * model = llama_model_load_from_file(model_path, params);
    llama_backend_free();
    return model == nullptr ? EXIT_SUCCESS : EXIT_FAILURE;
}
