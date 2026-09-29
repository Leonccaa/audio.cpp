# CUDA Q2_K row lookup regression

The RTTA branch is based on upstream `ed96b7307c8daba2ebcf7912af928825f6b14cb9`.
Upstream supplies the R2T2 incremental audio/KV cache. This branch adds CUDA
`GET_ROWS` support for Q2_K token embeddings found in mixed Q4_K_M files.
It does not change model weights, quantization, or request scheduling.

`test_q2_getrows.cpp` compares CPU and CUDA gathers for 256, 512, and 2048
columns, including repeated and out-of-order row indices. It checks 19,712
values. The September 29, 2026 CUDA 13 / RTX 5060 Ti run had zero maximum
absolute error in all cases. Link it against the built ggml, ggml-base,
ggml-cpu and ggml-cuda libraries, using the ggml include directory.

The reference calculation is `dequantize_block_q2_K` in `convert.cu`.
This is a local compatibility patch, not an upstream merged change.
Real-time requests still serialize within each loaded model/session.
