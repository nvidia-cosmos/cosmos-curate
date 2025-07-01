# Cosmos-Curate - End User Guide

- [Cosmos-Curate - End User Guide](#cosmos-curate---end-user-guide)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
      - [Hardware Requirement](#hardware-requirement)
      - [Software Requirement](#software-requirement)
      - [Additional Requirement](#additional-requirement)
  - [Initial Setup](#initial-setup)
  - [Quick Start for Local Run](#quick-start-for-local-run)
    - [Setup Environment and Install Dependencies](#setup-environment-and-install-dependencies)
    - [Run the Hello-World Example Pipeline](#run-the-hello-world-example-pipeline)
    - [Run the Reference Video Pipeline](#run-the-reference-video-pipeline)
    - [Useful Options for Local Run](#useful-options-for-local-run)
  - [Launch Pipelines on Slurm](#launch-pipelines-on-slurm)
    - [Prerequisites for Slurm Run](#prerequisites-for-slurm-run)
      - [Setup Password-less SSH to the Cluster](#setup-password-less-ssh-to-the-cluster)
      - [Identify User Path on the Cluster](#identify-user-path-on-the-cluster)
    - [Copy Config File, Cloud Storage Credential, and Model Files](#copy-config-file-cloud-storage-credential-and-model-files)
    - [Create sqsh Image and Copy to the Slurm Cluster](#create-sqsh-image-and-copy-to-the-slurm-cluster)
    - [Sync Source Code to the Slurm Cluster](#sync-source-code-to-the-slurm-cluster)
    - [Launch on Slurm](#launch-on-slurm)
    - [Find Logs](#find-logs)
  - [Launch Pipelines on NVIDIA DGX Cloud](#launch-pipelines-on-nvidia-dgx-cloud)
  - [Launch Pipelines on K8s Cluster (coming soon)](#launch-pipelines-on-k8s-cluster-coming-soon)
  - [Building the Client package](#building-the-client-package)
  - [Troubleshooting](#troubleshooting)
  - [Support](#support)
  - [Responsible Use of AI Models](#responsible-use-of-ai-models)

## Overview
Cosmos-Curate is a powerful tool for video curation and processing. This guide will help you get started with using the application.

## Prerequisites
#### Hardware Requirement
- Minimum 32GB host memory
- Minimum 200GB disk space
- One or more GPU with
  - minimum CUDA compute capability of 8.0
  - minimum memory of
    - 4GB, to run the hello-world pipeline
    - 48GB, to run the reference video pipelines

#### Software Requirement
- Ubuntu >= 22.04
- Python >=3.10 on your host
  - We will require a specific version in your virtual environment below.
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Note that the docker daemon needs to be restarted after the installation of NVIDIA Container Toolkit as described [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).

#### Additional Requirement
- [Hugging Face account and access token](https://huggingface.co/settings/tokens) (a read token should suffice for accessing the InternVideo2 model)
- [NGC API key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-api-key) for the NGC Registry to access the [NVIDIA-accelebrated container image for PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) (ensure your API key has the Private Registry permission)
- Cloud storage access credentials if using cloud storage
  - full support for S3-compatible object storage
  - basic support for Azure blob storage

## Initial Setup

1. Create a configuration file at `~/.config/cosmos_curate/config.yaml` and put your Hugging Face credentials:

```yaml
huggingface:
    user: "<your-username>"
    api_key: "<your-huggingface-token>"
```

2. To use InternVideo2 embedding model:
   - Visit [InternVideo2 Hugging Face page](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4/tree/main)
   - Log in to your Hugging Face account
   - Click "agree" to accept the model terms; this is required before you can download this model using HuggingFace API with your token

3. By default, `~/cosmos_curate_local_workspace/` is used as the local workspace for model weights and temporary files at runtime. To configure its location, set environment variable `COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX` to move it to `${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX}/cosmos_curate_local_workspace/`.
   - In other words, `"${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace"` is used as the local workspace.

4. Log into the NGC container registry via the Docker CLI. For the username, enter `'$oauthtoken'` exactly as shown. It is a special name that indicates that you will authenticate with an API key. Paste your key value at the password prompt.
```bash
docker login --username '$oauthtoken' nvcr.io
```

5. If you will run pipelines with videos on cloud storage, configure `~/.aws/credentials` properly.
   - All S3-compatible cloud storage should work with Cosmos-Curate.
     - Right now, since cosmos-curate only relies on `~/.aws/credentials` (not `~/.aws/config`), certain configuration entries need to be in `~/.aws/credentials`; e.g. `region` and `endpoint_url` if it's not AWS S3.
     - It should look similar to [this example file](../../examples/nvcf/creds/aws_credentials).
   - Azure blob storage is also supported but is tested much less extensively.
     - If using Azure blob storage, `~/.azure/credentials` should be configured properly.

## Quick Start for Local Run

The overall workflow is as follows:
1. Setup environment and install dependencies
   - This will give you a CLI for steps below
2. Build a docker container image
3. Download model weights from HuggingFace
4. Launch a pipeline
   - locally using local-docker launcher - **focus of this section**
   - on a slurm cluster using slurm launcer
   - on [NVIDIA Cloud Functions (NVCF)](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/overview.html) by reaching out to NVIDIA Cosmos-Curate team.
   - on Kubernetes cluster (coming soon)

### Setup Environment and Install Dependencies

- It is strongly recommended to use a Python virtual environment management system that can specify the Python version, such as
  [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html),
  [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html),
  [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), etc.
  - Also it is best to prevent packages under `$HOME/.local/` being used in the virtual environment,
    you can set environment variable **`export PYTHONNOUSERSITE=1`** to exclude user site-package directory.
- In case you are running in a headless display environment,
  you may need to set environment variable **`export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`**
  before running `poetry` to avoid getting stuck due to a keyring pop-up.

```bash
# 1. Create virtual environment (using micromamba as an example)
micromamba create -n cosmos-curate -c conda-forge python=3.10.18 poetry
micromamba activate cosmos-curate

# 2. Clone the repository and update `cosmos-xenna` submodule
git clone https://github.com/nvidia-cosmos/cosmos-curate.git
cd cosmos-curate
git submodule update --init

# 3. Install dependencies
pip install -e cosmos-xenna/
poetry install --extras=local

# 4. Verify the CLI tool is available
cosmos-curate --help
```

Alternatively, you may execute `./devset.sh` to complete inital setup of environment **from within your virtual environment**.

### Run the Hello-World Example Pipeline

The hello-world example pipeline aims to provide a minimal example to help understand the framework.
- Define the class for pipeline task as `HelloWorldTask` in [hello_world_pipeline.py](../../cosmos_curate/pipelines/examples/hello_world_pipeline.py).
- Define `GPT2` model in [gpt2](../../cosmos_curate/models/gpt2.py).
- Define 3 simple stages (`_LowerCaseStage`, `_PrintStage`, `_GPT2Stage`) in [hello_world_pipeline.py](../../cosmos_curate/pipelines/examples/hello_world_pipeline.py). So the functionality of this pipeline is:
  - stage 1: convert the input prompt in each `HelloWorldTask` to lower case;
  - stage 2: print the the converted input prompt;
  - stage 3: call GPT2 to generate some output;
- Call `cosmos_curate.core.interfaces.pipeline_interface.run_pipeline`.

```bash
# 1. Build a docker image for hello-world pipeline
#    - The hello-world pipeline uses the GPT-2 model
#    - We create a dedicated conda environment called transformers to run the GPT-2 model (hence `--env transformers`)
cosmos-curate image build --image-name cosmos-curate --image-tag hello-world --envs transformers

# 2. Download the GPT-2 model weights
cosmos-curate local launch --image-name cosmos-curate --image-tag hello-world -- python3 -m cosmos_curate.core.managers.model_cli download --models gpt2

# 3. Run the hellow-world pipeline
cosmos-curate local launch --image-name cosmos-curate --image-tag hello-world --curator-path . -- python3 -m cosmos_curate.pipelines.examples.hello_world_pipeline
```

### Run the Reference Video Pipeline

This section of the instructions references the concept of local paths. Note that these local paths are paths inside the container image, not paths on your local machine. Since `"${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace"` is mounted to `/config/` when launching the container, a path like `~/cosmos_curate_local_workspace/foo/` on your local machine needs to be specified as `/config/foo/` in the `cosmos-curate` commandline arguments.

1. **Build a docker image.**
   - Unlike the hello-world example, we run more than one models in this pipeline.
   - It's not always easy to run different models in the same python environment; so we need to build a new image with more conda environments included.
   - This could take up to 45 minutes for a fresh new build.

```bash
cosmos-curate image build --image-name cosmos-curate --image-tag 1.0.0 --envs text_curator,unified,video_splitting
```

2. **Download model weights from Hugging Face.**
   - For the same reason as above, we need to download weights for a few more models and it will take 10+ minutes depends on your network condition.

```bash
cosmos-curate local launch --image-name cosmos-curate --image-tag 1.0.0 -- python3 -m cosmos_curate.core.managers.model_cli download
```

3. **Run the Split-Annotate Pipeline**
   - **Input and output paths**
     - `--input-video-path` and `--output-clip-path` can be either a local path inside the container or an S3 path.
       - The input videos under `input_video_path` can have any directory hierarchy.
       - If your videos are under `~/cosmos_curate_local_workspace/raw_videos/` and you want the output to be under `~/cosmos_curate_local_workspace/output_clips/`, you should specify `--input-video-path /config/raw_videos/` and `--output-clip-path /config/output_clips/`.
     - When giving an S3 path, it need to start with `s3://`.
       - Right now, space is not allowed in a video's S3 path.
       - For example, you can give `--input-video-path s3://cosmos-curate-oss/raw_videos/` & `--output-clip-path s3://cosmos-curate-oss/output_clips/` to read from `s3://cosmos-curate-oss/raw_videos/` and write to `s3://cosmos-curate-oss/output_clips/`.
       - Please make sure your `~/.aws/credentials` file is configured properly as explained in [Initial Setup](#initial-setup) section above.
   - **`--limit` option**
     - `limit` specifies how many input videos under `input_video_path` to process.
     - Note when running locally with e.g. one GPU, a small `limit` value (like `1`) is needed to avoid running out of memory or disk.
   - **Failure recovery**
     - Failures are often inevitable due to hardware failures/glitches, kernel/driver/library bugs, etc., therefore this pipeline is carefully designed such that it can handle any crash gracefully without loss of compute time and a simple restart would resume from where it left resulting in correct behavior.

```bash
cosmos-curate local launch \
    --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
    -- python3 -m cosmos_curate.pipelines.video.run_pipeline split \
    --input-video-path <local or s3 path containing input videos> \
    --output-clip-path <local or s3 path to store output clips and metadatas> \
    --limit 1
```

At a high level, this pipeline
- splits each input video into shorter clips based on short transition
- transcodes each clip
- generates motion & aesthetic scores and filter clips (disabled by default)
- generates one descriptive caption for each 256-frame window in each clip
- stores the mp4 clips and metadatas to the specified `output_clip_path`

For more details, please refer to [Split-Annotate Pipeline](../curator/REFERENCE_PIPELINES_VIDEO.md#split-annotate-pipeline) section in [Reference Pipelines Guide](../curator/REFERENCE_PIPELINES_VIDEO.md).

4. **Optionally, Run the Split-Annotate Pipeline via API Endpoint**

First, launch the container with a service endpoint.

```bash
cosmos-curate local launch \
   --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
   -- python cosmos_curate/scripts/onto_nvcf.py --helm False
```

After `Application startup complete.` is printed in the log, you can invoke the split-annotate with a `curl` command.

```bash
curl -X POST http://localhost:8000/v1/run_pipeline -H "NVCF-REQID: 1234-5678" -d '{
    "pipeline": "split",
    "args": {
        "input_video_path": "<local or s3 path containing input videos>",
        "output_clip_path": "<local or s3 path to store output clips and metadatas>",
        "limit": 1
    }
}'
```

### Useful Options for Local Run

Almost all the CLI commands enable `no_args_is_help`, so running a command without any arguments will print out the help message.
For local launcher, you can simply run `cosmos-curate local launch` to see help messages for all the options.

A useful option is `--curator-path`; when this is given, the local launcher will mount the source code into the container,
such that you don't have to rebuild the container after code changes for local run.

## Launch Pipelines on Slurm

This section assumes that you have set up a local environment and have launched a pipeline locally.

### Prerequisites for Slurm Run

#### Setup Password-less SSH to the Cluster

Here are the [instructions](https://www.redhat.com/en/blog/passwordless-ssh) published on redhat.com.

Assume the login node of your Slurm cluster is `my-slurm-login-01.my-cluster.com`.
You can verify the password-less SSH setup by `ssh my-slurm-login-01.my-cluster.com`
and it should login directly without asking for password.

#### Identify User Path on the Cluster

Assume your user directory on the Slurm cluster is `/home/myusername/`. Note
- this path should be **accessible to all compute nodes**.
- this path should have enough disk quota to hold the image and model weights.

Set these environment variables on your local host.

```bash
export SLURM_USER_DIR="/home/myusername"
export SLURM_LOG_DIR="${SLURM_USER_DIR}/job_logs"
export SLURM_COSMOS_CURATE_CONFIG_DIR="${SLURM_USER_DIR}/.config/cosmos_curate"
export SLURM_AWS_CREDS_DIR="${SLURM_USER_DIR}/.aws"
export SLURM_WORKSPACE="${SLURM_USER_DIR}/cosmos_curate_local_workspace"
export SLURM_IMAGE_DIR="${SLURM_USER_DIR}/container_images"
export SLURM_SOURCE_DIR="${SLURM_USER_DIR}/src/cosmos-curate"
```


### Copy Config File, Cloud Storage Credential, and Model Files

```bash
# Copy ~/.config/cosmos_curate/config.yaml
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_COSMOS_CURATE_CONFIG_DIR}
rsync -avh ~/.config/cosmos_curate/config.yaml my-slurm-login-01.my-cluster.com:${SLURM_COSMOS_CURATE_CONFIG_DIR}/config.yaml

# Copy ~/.aws/credentials
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_AWS_CREDS_DIR}
rsync -avh ~/.aws/credentials my-slurm-login-01.my-cluster.com:${SLURM_AWS_CREDS_DIR}/credentials

# Copy models
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_WORKSPACE}/models
rsync -avh --progress ${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace/models/  my-slurm-login-01.my-cluster.com:${SLURM_WORKSPACE}/models/
```

### Create sqsh Image and Copy to the Slurm Cluster

1. Install `enroot` on your local machine based on the [instructions here](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md).

2. Import the hello world docker image built above to create a `.sqsh` file.

```bash
enroot import --output cosmos_curate+hello_world.sqsh dockerd://cosmos-curate:hello-world
```

3. Copy the sqsh file to slurm cluster.

```bash
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_IMAGE_DIR}
rsync -avh --progress ./cosmos_curate+hello_world.sqsh my-slurm-login-01.my-cluster.com:${SLURM_IMAGE_DIR}/
```

### Sync Source Code to the Slurm Cluster

```bash
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_SOURCE_DIR}/cosmos_curate/
rsync -avh ./cosmos_curate/ my-slurm-login-01.my-cluster.com:${SLURM_SOURCE_DIR}/cosmos_curate/
```

### Launch on Slurm

Figure out the following information for your Slurm cluster
- your user name on the slurm cluster, in case it is different than your local machine
- slurm account you are in
- slurm partition to use
- GRES, e.g. `gpu:4` or `gpu:8`

Configure the image path and container mounts

```bash
export CONTAINER_IMAGE="${SLURM_IMAGE_DIR}/cosmos_curate+hello_world.sqsh"

SLURM_AWS_CREDS_MOUNT="${SLURM_AWS_CREDS_DIR}/credentials:/creds/s3_creds"
SLURM_COSMOS_CURATE_CONFIG_MOUNT="${SLURM_COSMOS_CURATE_CONFIG_DIR}/config.yaml:/cosmos_curate/config/cosmos_curate.yaml"
SLURM_WORKSPACE_MOUNT="${SLURM_WORKSPACE}:/config"
SLURM_SOURCE_MOUNT="${SLURM_SOURCE_DIR}/cosmos_curate/:/opt/cosmos-curate/cosmos_curate"
export CONTAINER_MOUNTS="${SLURM_AWS_CREDS_MOUNT},${SLURM_COSMOS_CURATE_CONFIG_MOUNT},${SLURM_WORKSPACE_MOUNT},${SLURM_SOURCE_MOUNT}"
```

Launch!

```bash
cosmos-curate slurm submit \
  --login-node my-slurm-login-01.my-cluster.com \
  --username my_username_on_slurm_cluster \
  --account my_slurm_account \
  --partition my_slurm_gpu_partition \
  --job-name "hello-world" \
  --gres=my_slurm_cluster_gres \
  --num-nodes 1 \
  --remote-files-path "${SLURM_USER_DIR}/job_info" \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    -- python3 -m cosmos_curate.pipelines.examples.hello_world_pipeline
```

The command above will print the slurm job id like below
```bash
Submitted batch job <slurm_job_id>
```

### Find Logs

The slurm job log is at `"${SLURM_LOG_DIR}/{job_name}_{slurm_job_id}.log"` on the cluster.

You can also use the CLI to monitor the log:

```bash
cosmos-curate slurm job-log \
  --login-node my-slurm-login-01.my-cluster.com \
  --username my_username_on_slurm_cluster \
  --job-id slurm_job_id_printed_above
```

## Launch Pipelines on NVIDIA DGX Cloud

Cosmos-Curate can be deployed on [NVIDIA Cloud Function (NVCF)](https://docs.nvidia.com/cloud-functions/index.html) platform.

There are a few steps needed to get a new user onboarded to NVCF, so please reach out to NVIDIA Cosmos-Curate team and we will guide you through the process.

## Launch Pipelines on K8s Cluster (coming soon)

## Building the Client package
   - The `cosmos-curate` client can be built as a wheel and installed in a standalone mode, without the need for the rest of the source environment
```bash
poetry build
pip3 install dist/cosmos_curate*.whl
```

## Troubleshooting
If you encounter any issues:
1. Ensure your Hugging Face credentials are correctly configured
2. Verify that you have sufficient disk space for model downloads
3. Check that Docker is running and accessible
4. Check that [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed and `docker.service` is restarted after the installation of NVIDIA Container Toolkit.
6. Ensure you have the correct Python version installed

## Support
For additional support or to report issues, please contact the development team or create an issue in the repository. 

## Responsible Use of AI Models
[Responsible Use](./RESPONSIBLE_USE.md)

