#pragma once

#include "llama-impl.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

struct gguf_context;

static constexpr uint32_t LLAMA_REPACK_FORMAT_VERSION = 1;
static constexpr uint32_t LLAMA_REPACK_LAYOUT_VERSION = 1;
static constexpr size_t   LLAMA_REPACK_HEADER_SIZE     = 4096;

static constexpr const char * LLAMA_REPACK_KV_FORMAT_VERSION = "llama.repack.format_version";
static constexpr const char * LLAMA_REPACK_KV_LAYOUT_VERSION = "llama.repack.layout_version";
static constexpr const char * LLAMA_REPACK_KV_SOURCE_SHA256  = "llama.repack.source_sha256";
static constexpr const char * LLAMA_REPACK_KV_LAYOUTS        = "llama.repack.tensor_layouts";
static constexpr const char * LLAMA_REPACK_KV_CHECKSUMS      = "llama.repack.tensor_checksums";

struct llama_repack_header {
    uint64_t gguf_offset = LLAMA_REPACK_HEADER_SIZE;
    uint64_t gguf_meta_size = 0;
    uint64_t file_size = 0;
    uint64_t n_tensors = 0;
    uint64_t n_repacked = 0;
    std::array<uint8_t, 32> source_sha256 = {};
    std::array<uint8_t, 32> metadata_sha256 = {};
};

enum class llama_repack_probe {
    not_repack,
    valid,
    invalid,
};

llama_repack_probe llama_repack_read_header(
        const char * path,
        llama_repack_header & header,
        std::string & error);

bool llama_repack_write_header(
        FILE * file,
        const llama_repack_header & header,
        std::string & error);

bool llama_repack_validate_metadata(
        const llama_repack_header & header,
        const struct gguf_context * metadata,
        std::string & error);

std::array<uint8_t, 32> llama_repack_sha256(const void * data, size_t size);
std::string llama_repack_sha256_hex(const std::array<uint8_t, 32> & digest);
