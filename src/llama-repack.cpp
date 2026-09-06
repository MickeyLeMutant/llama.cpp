#include "llama-repack.h"

#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-mmap.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

extern "C" {
#include "hash/sha256/sha256.h"
}
#include "hash/xxhash/xxhash.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <system_error>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

static constexpr uint8_t REPACK_MAGIC[4] = { 'L', 'L', 'R', 'P' };
static constexpr size_t HEADER_DIGEST_OFFSET = 128;
static constexpr size_t HEADER_DIGEST_SIZE = 32;

#if defined(_WIN32)
#define LLAMA_REPACK_FSEEK _fseeki64
#define LLAMA_REPACK_FTELL _ftelli64
#else
#define LLAMA_REPACK_FSEEK fseeko
#define LLAMA_REPACK_FTELL ftello
#endif

void put_u32(uint8_t * dst, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        dst[i] = (uint8_t) (value >> (i * 8));
    }
}

void put_u64(uint8_t * dst, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        dst[i] = (uint8_t) (value >> (i * 8));
    }
}

uint32_t get_u32(const uint8_t * src) {
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        value |= (uint32_t) src[i] << (i * 8);
    }
    return value;
}

uint64_t get_u64(const uint8_t * src) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= (uint64_t) src[i] << (i * 8);
    }
    return value;
}

std::array<uint8_t, LLAMA_REPACK_HEADER_SIZE> encode_header(const llama_repack_header & header) {
    std::array<uint8_t, LLAMA_REPACK_HEADER_SIZE> data = {};
    memcpy(data.data(), REPACK_MAGIC, sizeof(REPACK_MAGIC));
    put_u32(data.data() + 4, LLAMA_REPACK_FORMAT_VERSION);
    put_u32(data.data() + 8, LLAMA_REPACK_HEADER_SIZE);
    put_u32(data.data() + 12, LLAMA_REPACK_LAYOUT_VERSION);
    put_u32(data.data() + 16, 0);
    put_u64(data.data() + 24, header.gguf_offset);
    put_u64(data.data() + 32, header.gguf_meta_size);
    put_u64(data.data() + 40, header.file_size);
    put_u64(data.data() + 48, header.n_tensors);
    put_u64(data.data() + 56, header.n_repacked);
    memcpy(data.data() + 64, header.source_sha256.data(), header.source_sha256.size());
    memcpy(data.data() + 96, header.metadata_sha256.data(), header.metadata_sha256.size());
    const auto digest = llama_repack_sha256(data.data(), HEADER_DIGEST_OFFSET);
    memcpy(data.data() + HEADER_DIGEST_OFFSET, digest.data(), HEADER_DIGEST_SIZE);
    return data;
}

bool all_zero(const uint8_t * data, size_t size) {
    return std::all_of(data, data + size, [](uint8_t value) { return value == 0; });
}

struct cpu_repack_api {
    ggml_backend_cpu_repack_get_layout_t get_layout = nullptr;
    ggml_backend_cpu_repack_get_rows_t get_rows = nullptr;
    ggml_backend_cpu_repack_tensor_t repack = nullptr;

    bool load(std::string & error) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        auto * cpu_reg = cpu_dev ? ggml_backend_dev_backend_reg(cpu_dev) : nullptr;
        if (cpu_reg == nullptr) {
            error = "CPU backend is not loaded";
            return false;
        }
        get_layout = (ggml_backend_cpu_repack_get_layout_t)
                ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_repack_get_layout");
        get_rows = (ggml_backend_cpu_repack_get_rows_t)
                ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_repack_get_rows");
        repack = (ggml_backend_cpu_repack_tensor_t)
                ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_repack_tensor");
        if (get_layout == nullptr || get_rows == nullptr || repack == nullptr) {
            error = "CPU backend was built without persistent repack support";
            return false;
        }
        return true;
    }
};

std::vector<std::string> source_paths(const char * path_source, const std::vector<std::string> & splits) {
    return splits.empty() ? std::vector<std::string> { path_source } : splits;
}

bool hash_source_paths(
        const std::vector<std::string> & paths,
        std::array<uint8_t, 32> & digest,
        std::string & error) {
    sha256_t state;
    sha256_init(&state);
    std::vector<uint8_t> buffer(4 * 1024 * 1024);
    for (const auto & path : paths) {
        try {
            llama_file file(path.c_str(), "rb");
            const uint64_t file_size = file.size();
            uint8_t size_bytes[8];
            put_u64(size_bytes, file_size);
            sha256_update(&state, size_bytes, sizeof(size_bytes));
            for (size_t offset = 0; offset < file.size();) {
                const size_t size = std::min(buffer.size(), file.size() - offset);
                file.read_raw(buffer.data(), size);
                sha256_update(&state, buffer.data(), size);
                offset += size;
            }
        } catch (const std::exception & exception) {
            error = format("failed to fingerprint source '%s': %s", path.c_str(), exception.what());
            return false;
        }
    }
    sha256_final(&state, digest.data());
    return true;
}

bool open_repack_metadata(
        const char * path,
        llama_repack_header & header,
        gguf_context_ptr & metadata,
        ggml_context ** tensor_context,
        std::string & error) {
    const llama_repack_probe probe = llama_repack_read_header(path, header, error);
    if (probe != llama_repack_probe::valid) {
        if (probe == llama_repack_probe::not_repack) {
            error = "file is not a persistent repack container";
        }
        return false;
    }

    FILE * file = ggml_fopen(path, "rb");
    if (file == nullptr || LLAMA_REPACK_FSEEK(file, header.gguf_offset, SEEK_SET) != 0) {
        if (file != nullptr) {
            fclose(file);
        }
        error = format("failed to open embedded GGUF in '%s'", path);
        return false;
    }
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ tensor_context,
    };
    metadata.reset(gguf_init_from_file_ptr(file, params));
    fclose(file);
    return metadata && llama_repack_validate_metadata(header, metadata.get(), error);
}

bool validate_tensor_checksums(
        const char * path,
        const llama_repack_header & header,
        const gguf_context * metadata,
        std::string & error) {
    const int key_checksums = gguf_find_key(metadata, LLAMA_REPACK_KV_CHECKSUMS);
    const auto * checksums = static_cast<const uint64_t *>(gguf_get_arr_data(metadata, key_checksums));
    std::vector<uint8_t> buffer(4 * 1024 * 1024);

    try {
        llama_file file(path, "rb");
        for (int64_t index = 0; index < gguf_get_n_tensors(metadata); ++index) {
            const char * name = gguf_get_tensor_name(metadata, index);
            // The embedded GGUF data offset is absolute because parsing started at its FILE position.
            const size_t offset = gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, index);
            const size_t size = gguf_get_tensor_size(metadata, index);
            XXH64_state_t * state = XXH64_createState();
            if (state == nullptr || XXH64_reset(state, 0) != XXH_OK) {
                XXH64_freeState(state);
                error = "failed to initialize tensor checksum state";
                return false;
            }
            file.seek(offset, SEEK_SET);
            for (size_t done = 0; done < size;) {
                const size_t chunk = std::min(buffer.size(), size - done);
                file.read_raw(buffer.data(), chunk);
                XXH64_update(state, buffer.data(), chunk);
                done += chunk;
            }
            const uint64_t actual = XXH64_digest(state);
            XXH64_freeState(state);
            if (actual != checksums[index]) {
                error = format("persistent repack checksum mismatch for tensor '%s'", name);
                return false;
            }
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }
    return true;
}

bool sync_file(FILE * file, std::string & error) {
    if (fflush(file) != 0) {
        error = format("failed to flush persistent repack file: %s", strerror(errno));
        return false;
    }
#if defined(_WIN32)
    if (_commit(_fileno(file)) != 0) {
#else
    if (fsync(fileno(file)) != 0) {
#endif
        error = format("failed to synchronize persistent repack file: %s", strerror(errno));
        return false;
    }
    return true;
}

bool atomic_replace(const std::string & temporary, const std::string & destination, bool force, std::string & error) {
#if defined(_WIN32)
    const std::filesystem::path temporary_path(temporary);
    const std::filesystem::path destination_path(destination);
    const DWORD flags = MOVEFILE_WRITE_THROUGH | (force ? MOVEFILE_REPLACE_EXISTING : 0);
    if (!MoveFileExW(temporary_path.c_str(), destination_path.c_str(), flags)) {
        error = format("failed to install persistent repack file (Windows error %lu)", GetLastError());
        return false;
    }
#else
    if (force && rename(temporary.c_str(), destination.c_str()) != 0) {
        error = format("failed to install persistent repack file: %s", strerror(errno));
        return false;
    }
    if (!force) {
        if (link(temporary.c_str(), destination.c_str()) != 0) {
            error = format("failed to install persistent repack file: %s", strerror(errno));
            return false;
        }
        if (unlink(temporary.c_str()) != 0) {
            error = format("persistent repack was installed, but its temporary link could not be removed: %s", strerror(errno));
            return false;
        }
    }
#endif
    return true;
}

} // namespace

std::array<uint8_t, 32> llama_repack_sha256(const void * data, size_t size) {
    std::array<uint8_t, 32> digest;
    sha256_t state;
    sha256_init(&state);
    sha256_update(&state, static_cast<const unsigned char *>(data), size);
    sha256_final(&state, digest.data());
    return digest;
}

std::string llama_repack_sha256_hex(const std::array<uint8_t, 32> & digest) {
    static constexpr char HEX[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        result[2 * i] = HEX[digest[i] >> 4];
        result[2 * i + 1] = HEX[digest[i] & 0x0f];
    }
    return result;
}

llama_repack_probe llama_repack_read_header(
        const char * path,
        llama_repack_header & header,
        std::string & error) {
    FILE * file = ggml_fopen(path, "rb");
    if (file == nullptr) {
        error = format("failed to open '%s': %s", path, strerror(errno));
        return llama_repack_probe::invalid;
    }

    std::array<uint8_t, LLAMA_REPACK_HEADER_SIZE> data = {};
    const size_t nread = fread(data.data(), 1, data.size(), file);
    if (nread < sizeof(REPACK_MAGIC) || memcmp(data.data(), REPACK_MAGIC, sizeof(REPACK_MAGIC)) != 0) {
        fclose(file);
        return llama_repack_probe::not_repack;
    }
    if (nread != data.size()) {
        fclose(file);
        error = "truncated persistent repack header";
        return llama_repack_probe::invalid;
    }

    if (LLAMA_REPACK_FSEEK(file, 0, SEEK_END) != 0) {
        fclose(file);
        error = "failed to determine persistent repack file size";
        return llama_repack_probe::invalid;
    }
    const int64_t actual_size = (int64_t) LLAMA_REPACK_FTELL(file);
    fclose(file);

    const uint32_t format_version = get_u32(data.data() + 4);
    const uint32_t header_size = get_u32(data.data() + 8);
    const uint32_t layout_version = get_u32(data.data() + 12);
    if (format_version != LLAMA_REPACK_FORMAT_VERSION) {
        error = format("unsupported persistent repack format version %u", format_version);
        return llama_repack_probe::invalid;
    }
    if (layout_version != LLAMA_REPACK_LAYOUT_VERSION) {
        error = format("unsupported persistent repack layout version %u", layout_version);
        return llama_repack_probe::invalid;
    }
    if (header_size != LLAMA_REPACK_HEADER_SIZE || get_u64(data.data() + 24) != LLAMA_REPACK_HEADER_SIZE) {
        error = "invalid persistent repack header or GGUF offset";
        return llama_repack_probe::invalid;
    }
    if (get_u32(data.data() + 16) != 0 || get_u32(data.data() + 20) != 0) {
        error = "persistent repack header has unsupported flags";
        return llama_repack_probe::invalid;
    }
    if (!all_zero(data.data() + 160, data.size() - 160)) {
        error = "persistent repack header has nonzero reserved bytes";
        return llama_repack_probe::invalid;
    }

    const auto expected_digest = llama_repack_sha256(data.data(), HEADER_DIGEST_OFFSET);
    if (memcmp(expected_digest.data(), data.data() + HEADER_DIGEST_OFFSET, HEADER_DIGEST_SIZE) != 0) {
        error = "persistent repack header checksum mismatch";
        return llama_repack_probe::invalid;
    }

    header.gguf_offset = get_u64(data.data() + 24);
    header.gguf_meta_size = get_u64(data.data() + 32);
    header.file_size = get_u64(data.data() + 40);
    header.n_tensors = get_u64(data.data() + 48);
    header.n_repacked = get_u64(data.data() + 56);
    memcpy(header.source_sha256.data(), data.data() + 64, header.source_sha256.size());
    memcpy(header.metadata_sha256.data(), data.data() + 96, header.metadata_sha256.size());

    if (actual_size < 0 || header.file_size != (uint64_t) actual_size ||
            header.gguf_meta_size == 0 || header.gguf_meta_size > header.file_size - header.gguf_offset) {
        error = "persistent repack file size or metadata bounds mismatch";
        return llama_repack_probe::invalid;
    }
    if (header.n_repacked > header.n_tensors) {
        error = "persistent repack tensor counts are invalid";
        return llama_repack_probe::invalid;
    }
    return llama_repack_probe::valid;
}

bool llama_repack_write_header(
        FILE * file,
        const llama_repack_header & header,
        std::string & error) {
    const auto data = encode_header(header);
    if (LLAMA_REPACK_FSEEK(file, 0, SEEK_SET) != 0 || fwrite(data.data(), 1, data.size(), file) != data.size()) {
        error = format("failed to write persistent repack header: %s", strerror(errno));
        return false;
    }
    return true;
}

bool llama_repack_validate_metadata(
        const llama_repack_header & header,
        const struct gguf_context * metadata,
        std::string & error) {
    const size_t metadata_size = gguf_get_meta_size(metadata);
    if (metadata_size != header.gguf_meta_size) {
        error = format("embedded GGUF metadata size mismatch: expected %zu, got %zu",
                (size_t) header.gguf_meta_size, metadata_size);
        return false;
    }

    std::vector<uint8_t> data(metadata_size);
    gguf_get_meta_data(metadata, data.data());
    const auto digest = llama_repack_sha256(data.data(), data.size());
    if (digest != header.metadata_sha256) {
        error = "embedded GGUF metadata checksum mismatch";
        return false;
    }

    const int key_format = gguf_find_key(metadata, LLAMA_REPACK_KV_FORMAT_VERSION);
    const int key_layout = gguf_find_key(metadata, LLAMA_REPACK_KV_LAYOUT_VERSION);
    const int key_source = gguf_find_key(metadata, LLAMA_REPACK_KV_SOURCE_SHA256);
    const int key_layouts = gguf_find_key(metadata, LLAMA_REPACK_KV_LAYOUTS);
    const int key_checksums = gguf_find_key(metadata, LLAMA_REPACK_KV_CHECKSUMS);
    if (key_format < 0 || key_layout < 0 || key_source < 0 || key_layouts < 0 || key_checksums < 0) {
        error = "embedded GGUF is missing persistent repack metadata";
        return false;
    }
    if (gguf_get_val_u32(metadata, key_format) != LLAMA_REPACK_FORMAT_VERSION ||
            gguf_get_val_u32(metadata, key_layout) != LLAMA_REPACK_LAYOUT_VERSION) {
        error = "embedded GGUF persistent repack version mismatch";
        return false;
    }
    if (llama_repack_sha256_hex(header.source_sha256) != gguf_get_val_str(metadata, key_source)) {
        error = "persistent repack source fingerprint mismatch";
        return false;
    }
    if (gguf_get_arr_type(metadata, key_layouts) != GGUF_TYPE_UINT32 ||
            gguf_get_arr_n(metadata, key_layouts) != header.n_tensors ||
            gguf_get_arr_type(metadata, key_checksums) != GGUF_TYPE_UINT64 ||
            gguf_get_arr_n(metadata, key_checksums) != header.n_tensors ||
            gguf_get_n_tensors(metadata) != (int64_t) header.n_tensors) {
        error = "persistent repack tensor manifest has the wrong type or length";
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(metadata);
    if (data_offset != header.gguf_offset + header.gguf_meta_size) {
        error = "persistent repack tensor data offset is invalid";
        return false;
    }
    const size_t alignment = gguf_get_alignment(metadata);
    if (alignment == 0 || data_offset % alignment != 0) {
        error = "persistent repack tensor alignment is invalid";
        return false;
    }

    const auto * layouts = static_cast<const uint32_t *>(gguf_get_arr_data(metadata, key_layouts));
    uint64_t n_repacked = 0;
    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(header.n_tensors);
    for (int64_t index = 0; index < gguf_get_n_tensors(metadata); ++index) {
        const size_t tensor_offset = gguf_get_tensor_offset(metadata, index);
        const size_t tensor_size = gguf_get_tensor_size(metadata, index);
        if (tensor_offset % alignment != 0 || tensor_offset > header.file_size - data_offset ||
                tensor_size > header.file_size - data_offset - tensor_offset) {
            error = format("persistent repack tensor '%s' has invalid bounds", gguf_get_tensor_name(metadata, index));
            return false;
        }
        ranges.emplace_back(data_offset + tensor_offset, data_offset + tensor_offset + tensor_size);
        n_repacked += layouts[index] != 0;
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index - 1].second > ranges[index].first) {
            error = "persistent repack tensor ranges overlap";
            return false;
        }
    }
    if (n_repacked != header.n_repacked) {
        error = "persistent repack tensor count does not match its manifest";
        return false;
    }
    return true;
}

struct llama_model_repack_params llama_model_repack_default_params(void) {
    struct llama_model_repack_params result = {
        /*.max_buffer_size = */ 64 * 1024 * 1024,
        /*.force           = */ false,
        /*.delete_source   = */ false,
    };
    return result;
}

int llama_model_repack_validate_file(
        const char * path_repack,
        const char * path_source,
        struct llama_model_params model_params,
        bool check_data) {
    std::string error;
    llama_repack_header header;
    gguf_context_ptr metadata;
    ggml_context * tensor_context_raw = nullptr;
    if (!open_repack_metadata(path_repack, header, metadata, &tensor_context_raw, error)) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.c_str());
        return -1;
    }
    ggml_context_ptr tensor_context(tensor_context_raw);

    model_params.no_alloc = true;
    model_params.check_tensors = false;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file(path_repack, model_params);
    if (model == nullptr) {
        LLAMA_LOG_ERROR("%s: persistent repack placement profile is incompatible\n", __func__);
        return -1;
    }
    llama_model_free(model);

    if (check_data && !validate_tensor_checksums(path_repack, header, metadata.get(), error)) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.c_str());
        return -1;
    }

    if (path_source != nullptr) {
        std::vector<std::string> splits;
        try {
            llama_model_loader source(nullptr, nullptr, nullptr, path_source, splits, nullptr,
                    LLAMA_LOAD_MODE_NONE, false, true, model_params.load_mtp,
                    model_params.kv_overrides, model_params.tensor_buft_overrides);
            std::array<uint8_t, 32> source_digest;
            if (!hash_source_paths(source_paths(path_source, splits), source_digest, error)) {
                LLAMA_LOG_ERROR("%s: %s\n", __func__, error.c_str());
                return -1;
            }
            if (source_digest != header.source_sha256) {
                LLAMA_LOG_ERROR("%s: source model fingerprint does not match '%s'\n", __func__, path_repack);
                return -1;
            }
        } catch (const std::exception & exception) {
            LLAMA_LOG_ERROR("%s: failed to validate source model: %s\n", __func__, exception.what());
            return -1;
        }
    }

    return 0;
}

int llama_model_repack_to_file(
        const char * path_source,
        const char * path_repack,
        struct llama_model_params model_params,
        struct llama_model_repack_params repack_params) {
    if (path_source == nullptr || path_repack == nullptr || path_source[0] == '\0' || path_repack[0] == '\0') {
        LLAMA_LOG_ERROR("%s: source and destination paths are required\n", __func__);
        return -1;
    }

    std::error_code filesystem_error;
    const std::filesystem::path source_path(path_source);
    const std::filesystem::path destination_path(path_repack);
    if (std::filesystem::equivalent(source_path, destination_path, filesystem_error)) {
        LLAMA_LOG_ERROR("%s: source and destination must be different files\n", __func__);
        return -1;
    }
    filesystem_error.clear();
    if (std::filesystem::exists(destination_path, filesystem_error) && !repack_params.force) {
        LLAMA_LOG_ERROR("%s: destination '%s' already exists; use force to replace it\n", __func__, path_repack);
        return -1;
    }

    std::string error;
    llama_repack_header existing_header;
    const llama_repack_probe source_probe = llama_repack_read_header(path_source, existing_header, error);
    if (source_probe != llama_repack_probe::not_repack) {
        LLAMA_LOG_ERROR("%s: source must be a canonical GGUF model%s%s\n", __func__,
                error.empty() ? "" : ": ", error.c_str());
        return -1;
    }

    cpu_repack_api cpu_api;
    if (!cpu_api.load(error)) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.c_str());
        return -1;
    }

    std::vector<std::string> splits;
    std::unique_ptr<llama_model_loader> source;
    try {
        source.reset(new llama_model_loader(nullptr, nullptr, nullptr, path_source, splits, nullptr,
                LLAMA_LOAD_MODE_NONE, false, true, model_params.load_mtp,
                model_params.kv_overrides, model_params.tensor_buft_overrides));
    } catch (const std::exception & exception) {
        LLAMA_LOG_ERROR("%s: failed to open source model: %s\n", __func__, exception.what());
        return -1;
    }

    std::array<uint8_t, 32> source_digest;
    const auto paths = source_paths(path_source, splits);
    for (const auto & path : paths) {
        filesystem_error.clear();
        if (std::filesystem::equivalent(std::filesystem::path(path), destination_path, filesystem_error)) {
            LLAMA_LOG_ERROR("%s: destination must not replace a source GGUF split\n", __func__);
            return -1;
        }
    }
    if (!hash_source_paths(paths, source_digest, error)) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.c_str());
        return -1;
    }

    struct llama_model_params plan_params = model_params;
    plan_params.no_alloc = true;
    plan_params.check_tensors = false;
    plan_params.load_mode = LLAMA_LOAD_MODE_NONE;
    std::unique_ptr<llama_model> plan(llama_model_load_from_file(path_source, plan_params));
    if (!plan) {
        LLAMA_LOG_ERROR("%s: failed to build tensor placement profile\n", __func__);
        return -1;
    }

    gguf_context_ptr output(gguf_init_empty());
    gguf_set_kv(output.get(), source->metadata);
    gguf_remove_key(output.get(), source->llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(output.get(), source->llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(output.get(), source->llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    std::vector<const llama_model_loader::llama_tensor_weight *> weights;
    weights.reserve(source->weights_map.size());
    std::vector<uint32_t> layouts;
    layouts.reserve(source->weights_map.size());
    uint64_t n_repacked = 0;
    size_t output_alignment = gguf_get_alignment(source->metadata);
    for (const auto & item : source->weights_map) {
        const auto * weight = &item.second;
        weights.push_back(weight);

        const ggml_tensor * planned = plan->get_tensor(item.first.c_str());
        if (planned == nullptr || planned->buffer == nullptr) {
            LLAMA_LOG_ERROR("%s: tensor '%s' is missing from placement profile\n", __func__, item.first.c_str());
            return -1;
        }
        const auto planned_buft = ggml_backend_buffer_get_type(planned->buffer);
        const bool selected_repack = strcmp(ggml_backend_buft_name(planned_buft), "CPU_REPACK") == 0;
        const uint32_t layout = selected_repack ? cpu_api.get_layout(weight->tensor) : 0;
        if (selected_repack && layout == 0) {
            LLAMA_LOG_ERROR("%s: CPU_REPACK selected without a layout for tensor '%s'\n", __func__, item.first.c_str());
            return -1;
        }
        layouts.push_back(layout);
        n_repacked += layout != 0;
        if (selected_repack) {
            output_alignment = std::max(output_alignment, ggml_backend_buft_get_alignment(planned_buft));
        }
    }
    if (n_repacked == 0) {
        LLAMA_LOG_ERROR("%s: current placement profile does not select any CPU_REPACK tensors\n", __func__);
        return -1;
    }

    std::vector<uint64_t> checksums(weights.size(), 0);
    gguf_set_val_u32(output.get(), GGUF_KEY_GENERAL_ALIGNMENT, (uint32_t) output_alignment);
    for (const auto * weight : weights) {
        gguf_add_tensor(output.get(), weight->tensor);
    }
    gguf_set_val_u32(output.get(), LLAMA_REPACK_KV_FORMAT_VERSION, LLAMA_REPACK_FORMAT_VERSION);
    gguf_set_val_u32(output.get(), LLAMA_REPACK_KV_LAYOUT_VERSION, LLAMA_REPACK_LAYOUT_VERSION);
    gguf_set_val_str(output.get(), LLAMA_REPACK_KV_SOURCE_SHA256, llama_repack_sha256_hex(source_digest).c_str());
    gguf_set_arr_data(output.get(), LLAMA_REPACK_KV_LAYOUTS, GGUF_TYPE_UINT32, layouts.data(), layouts.size());
    gguf_set_arr_data(output.get(), LLAMA_REPACK_KV_CHECKSUMS, GGUF_TYPE_UINT64, checksums.data(), checksums.size());

    size_t tensor_data_size = 0;
    for (size_t index = 0; index < weights.size(); ++index) {
        const size_t offset = gguf_get_tensor_offset(output.get(), index);
        const size_t size = ggml_nbytes(weights[index]->tensor);
        if (offset > std::numeric_limits<size_t>::max() - size) {
            LLAMA_LOG_ERROR("%s: persistent repack output size overflow\n", __func__);
            return -1;
        }
        tensor_data_size = std::max(tensor_data_size, offset + size);
    }
    const uintmax_t estimated_size = LLAMA_REPACK_HEADER_SIZE + gguf_get_meta_size(output.get()) + tensor_data_size;
    const std::filesystem::path destination_dir = destination_path.has_parent_path() ? destination_path.parent_path() : std::filesystem::current_path();
    filesystem_error.clear();
    const auto space = std::filesystem::space(destination_dir, filesystem_error);
    if (!filesystem_error && estimated_size > space.available) {
        LLAMA_LOG_ERROR("%s: persistent repack requires approximately %.2f GiB, but only %.2f GiB is available\n", __func__,
                (double) estimated_size / (1024.0 * 1024.0 * 1024.0),
                (double) space.available / (1024.0 * 1024.0 * 1024.0));
        return -1;
    }

    const size_t workspace_size = repack_params.max_buffer_size ? repack_params.max_buffer_size : 64 * 1024 * 1024;
#if defined(_WIN32)
    const int process_id = _getpid();
#else
    const int process_id = getpid();
#endif
    const std::string temporary = std::string(path_repack) + ".tmp." +
            std::to_string(process_id) + "." + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    FILE * file = ggml_fopen(temporary.c_str(), "w+b");
    if (file == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to create temporary file '%s': %s\n", __func__, temporary.c_str(), strerror(errno));
        return -1;
    }
    auto fail = [&](const std::string & message, int status = -1) {
        fclose(file);
        std::remove(temporary.c_str());
        LLAMA_LOG_ERROR("%s: %s\n", __func__, message.c_str());
        return status;
    };

    std::array<uint8_t, LLAMA_REPACK_HEADER_SIZE> empty_header = {};
    if (fwrite(empty_header.data(), 1, empty_header.size(), file) != empty_header.size() ||
            !gguf_write_to_file_ptr(output.get(), file, true)) {
        return fail("failed to write persistent repack metadata placeholder");
    }

    const size_t data_offset = gguf_get_meta_size(output.get());
    uint64_t bytes_done = 0;
    uint64_t bytes_total = 0;
    for (const auto * weight : weights) {
        bytes_total += ggml_nbytes(weight->tensor);
    }

    for (size_t index = 0; index < weights.size(); ++index) {
        const auto * weight = weights[index];
        const ggml_tensor * tensor = weight->tensor;
        const size_t tensor_size = ggml_nbytes(tensor);
        const size_t output_offset = LLAMA_REPACK_HEADER_SIZE + data_offset + gguf_get_tensor_offset(output.get(), index);
        if (LLAMA_REPACK_FSEEK(file, output_offset, SEEK_SET) != 0) {
            return fail(format("failed to seek output tensor '%s'", ggml_get_name(tensor)));
        }

        XXH64_state_t * checksum = XXH64_createState();
        if (checksum == nullptr || XXH64_reset(checksum, 0) != XXH_OK) {
            XXH64_freeState(checksum);
            return fail("failed to initialize tensor checksum");
        }

        const uint32_t layout = layouts[index];
        if (layout == 0) {
            const size_t chunk_size = std::max<size_t>(1, std::min(workspace_size, tensor_size));
            std::vector<uint8_t> buffer(chunk_size);
            source->files[weight->idx]->seek(weight->offs, SEEK_SET);
            for (size_t offset = 0; offset < tensor_size;) {
                const size_t size = std::min(buffer.size(), tensor_size - offset);
                source->files[weight->idx]->read_raw(buffer.data(), size);
                if (fwrite(buffer.data(), 1, size, file) != size) {
                    XXH64_freeState(checksum);
                    return fail(format("failed to write tensor '%s'", ggml_get_name(tensor)));
                }
                XXH64_update(checksum, buffer.data(), size);
                offset += size;
                bytes_done += size;
                if (model_params.progress_callback && !model_params.progress_callback(
                        (float) bytes_done / bytes_total, model_params.progress_callback_user_data)) {
                    XXH64_freeState(checksum);
                    return fail("persistent repack cancelled", -2);
                }
            }
        } else {
            const int64_t group_rows = cpu_api.get_rows(layout);
            const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            if (group_rows <= 0 || row_size == 0 || (size_t) group_rows > workspace_size / 2 / row_size) {
                XXH64_freeState(checksum);
                return fail(format("workspace is too small for one repack row group of tensor '%s'", ggml_get_name(tensor)));
            }
            int64_t chunk_rows = (int64_t) (workspace_size / 2 / row_size);
            chunk_rows -= chunk_rows % group_rows;
            std::vector<uint8_t> source_buffer(row_size * chunk_rows);
            std::vector<uint8_t> repack_buffer(row_size * chunk_rows);
            const int64_t planes = tensor->ne[2] * tensor->ne[3];
            for (int64_t plane = 0; plane < planes; ++plane) {
                for (int64_t row = 0; row < tensor->ne[1];) {
                    const int64_t rows = std::min(chunk_rows, tensor->ne[1] - row);
                    const size_t size = row_size * rows;
                    const size_t source_offset = weight->offs + (plane * tensor->ne[1] + row) * row_size;
                    source->files[weight->idx]->seek(source_offset, SEEK_SET);
                    source->files[weight->idx]->read_raw(source_buffer.data(), size);
                    if (cpu_api.repack(tensor, layout, source_buffer.data(), repack_buffer.data(), rows) != 0 ||
                            fwrite(repack_buffer.data(), 1, size, file) != size) {
                        XXH64_freeState(checksum);
                        return fail(format("failed to repack tensor '%s'", ggml_get_name(tensor)));
                    }
                    XXH64_update(checksum, repack_buffer.data(), size);
                    row += rows;
                    bytes_done += size;
                    if (model_params.progress_callback && !model_params.progress_callback(
                            (float) bytes_done / bytes_total, model_params.progress_callback_user_data)) {
                        XXH64_freeState(checksum);
                        return fail("persistent repack cancelled", -2);
                    }
                }
            }
        }
        checksums[index] = XXH64_digest(checksum);
        XXH64_freeState(checksum);
    }

    gguf_set_arr_data(output.get(), LLAMA_REPACK_KV_CHECKSUMS, GGUF_TYPE_UINT64, checksums.data(), checksums.size());
    const size_t final_metadata_size = gguf_get_meta_size(output.get());
    if (final_metadata_size != data_offset) {
        return fail("persistent repack metadata changed size during finalization");
    }
    std::vector<uint8_t> metadata_bytes(final_metadata_size);
    gguf_get_meta_data(output.get(), metadata_bytes.data());
    if (LLAMA_REPACK_FSEEK(file, LLAMA_REPACK_HEADER_SIZE, SEEK_SET) != 0 ||
            fwrite(metadata_bytes.data(), 1, metadata_bytes.size(), file) != metadata_bytes.size()) {
        return fail("failed to finalize persistent repack metadata");
    }

    if (LLAMA_REPACK_FSEEK(file, 0, SEEK_END) != 0) {
        return fail("failed to determine persistent repack output size");
    }
    llama_repack_header header;
    header.gguf_meta_size = final_metadata_size;
    header.file_size = LLAMA_REPACK_FTELL(file);
    header.n_tensors = weights.size();
    header.n_repacked = n_repacked;
    header.source_sha256 = source_digest;
    header.metadata_sha256 = llama_repack_sha256(metadata_bytes.data(), metadata_bytes.size());
    if (!llama_repack_write_header(file, header, error) || !sync_file(file, error)) {
        return fail(error);
    }
    fclose(file);
    file = nullptr;

    if (llama_model_repack_validate_file(temporary.c_str(), path_source, model_params, true) != 0) {
        std::remove(temporary.c_str());
        LLAMA_LOG_ERROR("%s: validation of temporary persistent repack file failed\n", __func__);
        return -1;
    }
    if (!atomic_replace(temporary, path_repack, repack_params.force, error)) {
        std::remove(temporary.c_str());
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.c_str());
        return -1;
    }
    if (llama_model_repack_validate_file(path_repack, nullptr, model_params, false) != 0) {
        LLAMA_LOG_ERROR("%s: installed persistent repack file failed validation; source was retained\n", __func__);
        return -1;
    }

    plan.reset();
    source.reset();

    if (repack_params.delete_source) {
        for (const auto & path : paths) {
            std::error_code remove_error;
            if (!std::filesystem::remove(std::filesystem::path(path), remove_error)) {
                LLAMA_LOG_ERROR("%s: persistent repack is valid, but failed to delete source '%s': %s\n",
                        __func__, path.c_str(), remove_error.message().c_str());
                return -1;
            }
        }
    }

    LLAMA_LOG_INFO("%s: wrote %" PRIu64 " tensors (%" PRIu64 " repacked) to %s\n",
            __func__, header.n_tensors, header.n_repacked, path_repack);
    return 0;
}
