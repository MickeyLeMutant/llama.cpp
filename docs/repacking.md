# Persistent CPU weight repacking

llama.cpp normally converts supported quantized CPU weights to an optimized layout when a model is loaded. Runtime repacking keeps the portable GGUF unchanged, but it allocates the converted weights in memory. `--no-repack` avoids that allocation and uses canonical GGUF weights with less optimized kernels.

Persistent repacking performs the same conversion once and stores the result in a self-contained `.repack` file. Later loads memory-map those bytes directly, without allocating another model-sized `CPU_REPACK` buffer.

Persistent repack files are not GGUF files. They have their own magic and format version so other GGUF readers cannot mistake optimized bytes for canonical quantized blocks.

## Create a repack file

Build llama.cpp with the normal tools enabled, then run:

```bash
llama-repack -m model.gguf -o model.repack
```

The source remains unchanged. An existing destination is not replaced unless `--force` is specified:

```bash
llama-repack -m model.gguf -o model.repack --force
```

To remove the source after a successful conversion:

```bash
llama-repack -m model.gguf -o model.repack --delete-source
```

The source is deleted only after the temporary output is written, synchronized, fully validated, atomically installed, and reopened successfully. For a split GGUF, all parts are combined into one repack file and all source parts are deleted only after validation succeeds.

Conversion reads and writes tensors in bounded chunks. Its default temporary workspace is 64 MiB, independent of total model size. The destination filesystem must temporarily have enough space for both the source and repack when `--delete-source` is used.

## Create or reuse a cache automatically

Model-loading tools that use the common arguments, including `llama-cli` and `llama-server`, accept:

```bash
llama-cli -m model.gguf --repack-cache model.repack
```

On a cache miss, llama.cpp creates the repack with streaming conversion and then loads it. On a hit, it validates the container version, source fingerprint, current CPU layout, and tensor placement profile before mapping the existing file. An invalid or incompatible cache is rebuilt from the source.

The source fingerprint covers every byte in every GGUF split. Cache validation therefore reads the source files when they are available.

## Load without the source GGUF

A repack file preserves the model metadata, tokenizer, logical tensor information, canonical tensors, and repacked tensors required for standalone loading:

```bash
llama-cli --repack-file model.repack
llama-server --repack-file model.repack
```

`--repack-file` does not require `-m`. Like `llama-repack`, direct loading uses a CPU-only placement when `--gpu-layers` is not specified. Use `--check-tensors` to verify all per-tensor checksums while loading. Header, metadata, bounds, layout compatibility, and placement compatibility are always checked.

## Placement and portability

Only tensors selected for the CPU repack buffer are stored in an optimized CPU layout. Tensors assigned to a GPU or another backend remain canonical. This supports profiles such as CPU-resident MoE experts with other layers on a GPU.

Use the same placement options when creating and loading the file, including `--gpu-layers`, `--cpu-moe`, split mode, device selection, tensor splits, and tensor buffer overrides. Direct loading rejects a profile that would interpret a canonical tensor as repacked or a repacked tensor as canonical.

Persistent files are specific to the selected CPU kernel layout and can be less portable than GGUF. Copy the original GGUF to different hardware, or regenerate the `.repack` file when llama.cpp reports a format, layout, CPU, backend, or placement incompatibility.

`--repack-cache` and `--repack-file` cannot be combined with `--no-repack`.
