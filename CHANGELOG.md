# Changelog

## Latest

## [1.1.8]

### Released
- 2025-11-17

### Added
- Nemotron-Nano-12B-v2-VL as an alternative VLM captioning model
- Gemini API as an option for video captioning
- Improved helm chart to simplify vanilla k8s deployment
- Upgraded cosmos-xenna to 0.1.7 for better scalability
- Significantly improved test coverage

### Fixed
- Fixed a bug in clip windowing utils which caused wrong caption for later windows within a clip
- Allow underscore in S3 bucket name
- Set cudagraph mode to piecewise for Qwen-based VL models to mitigate failure with illegal memory access
- Improved exception handling in vllm-captioning stage setup and process

### Documentation
- Added documentation for [vllm_interface](https://github.com/nvidia-cosmos/cosmos-curate/tree/main/docs/curator/VLLM_INTERFACE_DESIGN.md) which simplifies the integration of new vLLM-powered VLMs for captioning.

## [1.1.7]

### Released
- 2025-10-30

### Added
- Azure OpenAI API as an option to enhance captions
- Increased test coverage for vllm_interface to 100%
- Azure Blob Storage support for Slurm deployments
- Support multipart result zips
- Update python version to 3.10.19
- Retry vllm captioning on engine failure

### Fixed
- Switch torch package to pypi in unified
- Resolve hello_world pipeline execution with transformers 
- vLLM stage 2 captioning bug

## [1.1.6]

### Released
- 2025-10-16

### Added
- An example workflow script to operate X nvcf function to run M jobs
- Upgrade vllm to 0.11.0
- Upgrade transformers to 4.57.0
- Agent context files for Codex, Claude, and Gemini
- Runner abstraction for pipeline execution
- Increase test coverage

### Fixed
- Allow extra environment variables to be passed to the pixi runtime env
- Let slurm env setting override defaults inside container
- Remove dependency on pynvml
- Remove `max_seq_len_to_capture` from vLLM engine creation
- Improve the speed for final summary generation
- Downgrade click dep version to fix ray and revive e2e nvcf test

## [1.1.5]

### Released
- 2025-09-26

### Added
- Upgrade to [cosmos-xenna 0.1.6](https://pypi.org/project/cosmos-xenna/0.1.6/) for improved performance.

### Changed
- Update default parameters for stages' cpu resource requests for higher throughput

## [1.1.4]

### Released
- 2025-09-17

### Added

- Add [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) as an option for `EnhanceCaption` stage.
- Enable batching for internvideo2 embedding stage for improved throughput.
- Upgrade to [cosmos-xenna 0.1.5](https://pypi.org/project/cosmos-xenna/0.1.5/) for improved performance and stability.

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
