# Persistent CPU weight repacking

## Scope

Runtime CPU repacking improves matrix multiplication throughput, but it also requires an additional allocation as large as the affected model weights on every load. Persistent repacking moves that conversion to an explicit, one-time operation and maps the converted bytes directly on later loads.

The persistent file is intentionally not a GGUF file. Its first bytes are an `LLRP` header followed by a self-contained embedded GGUF metadata and tensor table. This prevents unmodified GGUF readers from interpreting repacked bytes as canonical quantized blocks.

## Compatibility contract

The outer header contains:

- a container format version;
- a CPU repack layout ABI version;
- the embedded GGUF offset and total file size;
- a source-model SHA-256 fingerprint;
- a SHA-256 digest of the embedded metadata.

The embedded metadata contains one stable layout identifier and one checksum for every tensor. Layout zero means canonical GGUF bytes. Nonzero identifiers describe exact CPU kernel layouts; they are not merely quantization types.

At load time, the normal placement policy is evaluated first. A nonzero layout is accepted only when the tensor is selected for `CPU_REPACK` and the current CPU backend selects the identical layout identifier. A canonical tensor is rejected when the current placement would require runtime repacking. This makes mixed CPU/GPU files safe: GPU tensors stay canonical, while only the CPU tensors selected by the saved placement profile are repacked.

Any change to the byte interpretation of an existing layout identifier must bump the layout ABI version. New layouts receive new identifiers.

## Write and load path

The writer reads source tensors in bounded row groups. Repacked tensors use the same backend conversion routine as runtime repacking; canonical tensors are copied. It writes a temporary file, finalizes checksums and metadata, flushes it, validates it, and atomically renames it. Source deletion, when requested, occurs only after the renamed file has passed validation and a metadata-only model load.

The loader maps the file and creates a non-owning `CPU_REPACK` backend buffer over the mapped range. Tensor initialization installs the normal CPU repack kernel traits, but tensor data is already in its final layout: no tensor-sized conversion allocation or copy is performed.

## Cache behavior

An automatic cache entry is reusable only when all of these match:

- container and layout ABI versions;
- source-model fingerprint;
- tensor placement and per-tensor layout profile;
- embedded metadata integrity.

An incompatible or incomplete cache is rebuilt through a new temporary file; it is never updated in place.
