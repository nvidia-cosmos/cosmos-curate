# Changelog


## Latest

## [1.1.3]

### Released
- 2025-09-08

### Added
- Release Grafana dashboard for pipeline monitoring.
- Add inflight batching for VLM captioning throughput.

### Changed
- Merge `video-splitting` env into `unified` env.
- Improve Slurm instructions.

## [1.1.2]

### Released
- 2025-08-28

### Added
- Upgrade to [cosmos-xenna 0.1.3](https://pypi.org/project/cosmos-xenna/0.1.3/) for improved scalability and observability.
- Enable Semantic Deduplication on Ray and improve IO efficiency for improved throughput.

## [1.1.1]

### Released
- 2025-08-13

### Added
- Add stage2 caption support to VLLMCaptionStage
- Add Nsight Systems for CUDA profiling

### Fixed
- Avoid unnecessary post-install docker layers
- Pin Ray to the same version for both pixi and poetry
- Update slurm cli to work with pixi

## [1.1.0]

### Released
- 2025-08-12

### Added
- Use [pixi](docs/DEVELOPER_GUIDE.md#working-with-pixi-environments) to manage environments inside container image
- Use absolute URL for [cosmos-xenna](https://github.com/nvidia-cosmos/cosmos-xenna) submodule; PLEASE run `git submodule sync` after pulling update
- Support for [Cosmos-Reason1](https://github.com/nvidia-cosmos/cosmos-reason1) as an alternative model for captioning
- Support for running [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) with [vLLM](https://docs.vllm.ai/en/latest/)

### Fixed
- Suppress warnings to make log more readable
- Make `/dev/shm` (and hence Ray object store) a fraction of system memory in local mode.

## [1.0.2]

### Released
- 2025-07-28

### Added
- Support for using multiple GPUs in captioning stage to enable large models
- Support for generating dataset to post-train [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world.md)
- Support for [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) as an alternative model for captioning

### Fixed
- PyNvCodec path for video decoding by fixing NVIDIA_DRIVER_CAPABILITIES env var
- CLI to import existing NVCF functions

## [1.0.1]

### Released
- 2025-07-11

### Added
- Multi-camera AV video split and caption pipelines
- Semantic-deduplication pipeline
- Support for [Cosmos-Embed1](https://research.nvidia.com/labs/dir/cosmos-embed1/) embedding model
- Support for using pre-signed URLs as input and output paths

### Fixed
- Splitting & transcoding accuracy for MPEG-TS files

### Changed
- Update required python version from 3.10.14 to 3.10.18

### Security
- Upgrade base image and packages to mitigate security vulnerabilities


## [1.0.0]

### Released
- 2025-06-11

### Added
- Initial version
