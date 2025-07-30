# Changelog


## Latest

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
