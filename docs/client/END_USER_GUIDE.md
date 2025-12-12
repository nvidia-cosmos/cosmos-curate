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
    - [Use Gemini for Captioning](#use-gemini-for-captioning)
    - [Enhance Captions with OpenAI](#enhance-captions-with-openai)
    - [Generate Dataset for Cosmos-Predict2 Post-Training](#generate-dataset-for-cosmos-predict2-post-training)
    - [Useful Options for Local Run](#useful-options-for-local-run)
  - [Launch Pipelines on Slurm](#launch-pipelines-on-slurm)
    - [Prerequisites for Slurm Run](#prerequisites-for-slurm-run)
      - [Setup Password-less SSH to the Cluster](#setup-password-less-ssh-to-the-cluster)
      - [Identify User Path on the Cluster](#identify-user-path-on-the-cluster)
    - [Copy Config File, Cloud Storage Credentials, and Model Files to Cluster](#copy-config-file-cloud-storage-credentials-and-model-files-to-cluster)
    - [Create sqsh Image and Copy to the Slurm Cluster](#create-sqsh-image-and-copy-to-the-slurm-cluster)
    - [Launch on Slurm](#launch-on-slurm)
    - [Find Logs](#find-logs)
    - [Developing on Slurm](#developing-on-slurm)
      - [Add the source-code mount to the `$CONTAINER_MOUNTS`:](#add-the-source-code-mount-to-the-container_mounts)
      - [Sync source code to the slurm cluster](#sync-source-code-to-the-slurm-cluster)
    - [Speeding up Model Load on Slurm](#speeding-up-model-load-on-slurm)
  - [Launch Pipelines on NVIDIA DGX Cloud](#launch-pipelines-on-nvidia-dgx-cloud)
  - [Launch Pipelines on K8s Cluster (coming soon)](#launch-pipelines-on-k8s-cluster-coming-soon)
  - [Observability for Pipelines](#observability-for-pipelines)
  - [Build the Client package](#build-the-client-package)
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
- Python >=3.12 on your host
  - We will require a specific version in your virtual environment below.
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Note that the docker daemon needs to be restarted after the installation of NVIDIA Container Toolkit as described [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).

#### Additional Requirement
- [Hugging Face account and access token](https://huggingface.co/settings/tokens) (a read token should suffice for accessing the InternVideo2 model)
- [NGC API key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-api-key) for the NGC Registry to access the [NVIDIA CUDA container image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) (your API key should at least have catalog permission)
- Cloud storage access credentials if using cloud storage
  - full support for S3-compatible object storage
  - basic support for Azure blob storage

## Initial Setup

1. Create a configuration file at `~/.config/cosmos_curate/config.yaml` and put your credentials. The Hugging Face section is required for model downloads; the Gemini and OpenAI sections are optional but needed for their respective captioning features:

```yaml
huggingface:
    user: "<your-username>"
    api_key: "<your-huggingface-token>"
gemini:
    api_key: "<your-gemini-api-key>"
openai:
    api_key: "<your-openai-api-key>"
    base_url: "https://<optional-base-url>/v1"
```

2. To use `InternVideo2` embedding model:
   - Visit [InternVideo2 Hugging Face page](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4/tree/main)
   - Log in to your Hugging Face account
   - Click "agree" to accept the model terms; this is required before you can download this model using HuggingFace API with your token

3. By default, `~/cosmos_curate_local_workspace/` is used as the local workspace for model weights and temporary files at runtime. To configure its location, set environment variable `COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX` to move it to `${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX}/cosmos_curate_local_workspace/`.
   - In other words, `"${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace"` is used as the local workspace.

4. Log into the NGC container registry via the Docker CLI. For the username, use `'$oauthtoken'` exactly as shown. It is a special name that indicates that you will authenticate with an API key. Paste your key value at the password prompt.
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

**The overall workflow is as follows:**
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
  [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
  [uv](https://docs.astral.sh/uv/),
  etc.
  - Also it is best to prevent packages under `$HOME/.local/` being used in the virtual environment,
    you can set environment variable **`export PYTHONNOUSERSITE=1`** to exclude user site-package directory.
- In case you are running in a headless display environment,
  you may need to set environment variable **`export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`**
  before running `poetry` to avoid getting stuck due to a keyring pop-up.

```bash
# 1. Create virtual environment (using micromamba as an example)
micromamba create -n cosmos-curate -c conda-forge python=3.12.12 poetry
micromamba activate cosmos-curate

# 2. Clone the repository and update `cosmos-xenna` submodule
git clone --recurse-submodules https://github.com/nvidia-cosmos/cosmos-curate.git
cd cosmos-curate

# 3. Install dependencies
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

There is a detailed walk-through in [Pipeline Design Guide](../curator/PIPELINE_DESIGN_GUIDE.md) to help understand how to build a pipeline.
The steps below only shows how to run the pipeline.

```bash
# 1. Build a docker image for hello-world pipeline
#    - The hello-world pipeline uses the GPT-2 model
#    - We create a dedicated conda environment called transformers to run the GPT-2 model (hence `--env transformers`)
cosmos-curate image build --image-name cosmos-curate --image-tag hello-world --envs transformers

# 2. Download the GPT-2 model weights
cosmos-curate local launch --image-name cosmos-curate --image-tag hello-world -- pixi run python -m cosmos_curate.core.managers.model_cli download --models gpt2

# 3. Run the hello-world pipeline
cosmos-curate local launch --image-name cosmos-curate --image-tag hello-world --curator-path . -- pixi run python -m cosmos_curate.pipelines.examples.hello_world_pipeline
```

### Run the Reference Video Pipeline

This section of the instructions references the concept of local paths. Note that these local paths are paths inside the container image, not paths on your local machine. Since `"${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace"` is mounted to `/config/` when launching the container, a path like `~/cosmos_curate_local_workspace/foo/` on your local machine needs to be specified as `/config/foo/` in the `cosmos-curate` commandline arguments.

1. **Build a docker image.**
   - Unlike the hello-world example, we run more than one models in this pipeline.
   - It's not always easy to run different models in the same python environment; so we need to build a new image with more conda environments included.
   - This could take up to 30 minutes for a fresh new build.

```bash
cosmos-curate image build --image-name cosmos-curate --image-tag 1.0.0
```

2. **Download model weights from Hugging Face.**
   - For the same reason as above, we need to download weights for a few more models and it will take 10+ minutes depends on your network condition.

```bash
cosmos-curate local launch --image-name cosmos-curate --image-tag 1.0.0 -- pixi run python -m cosmos_curate.core.managers.model_cli download
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
    -- pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
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

### Use Gemini for Captioning

Cosmos-Curate can call the Google Gemini API instead of local captioning models. To enable it:

1. Add your Gemini API key to `~/.config/cosmos_curate/config.yaml` under the `gemini` section as shown in [Initial Setup](#initial-setup). The key must also be accessible inside the container (the config file is mounted automatically when you use `--curator-path .`).
2. Select the Gemini captioning algorithm when launching the pipeline. The example below also increases `--captioning-max-output-tokens` to `4096`, which avoids Gemini truncation and has worked well in practice:

```bash
cosmos-curate local launch \
    --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
    -- pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
    --input-video-path <input path> \
    --output-clip-path <output path> \
    --captioning-algorithm gemini \
    --captioning-max-output-tokens 4096 \
    --gemini-model-name models/gemini-2.5-pro
```

You can further tune the behaviour with:
- `--gemini-caption-retries` / `--gemini-retry-delay-seconds` to adjust retry policy.
- `--gemini-max-inline-mb` to cap the inline MP4 size sent to Gemini (default `20.0` MB).

If Gemini returns block reasons or empty responses, the stage will surface those details in the clip errors.

### Enhance Captions with OpenAI

For a second-pass refinement of captions you can call the OpenAI API.

1. Populate the `openai` section in `~/.config/cosmos_curate/config.yaml` with your API key (and optional `base_url`).
2. Launch the pipeline with the enhance caption stage enabled and point it to your model:

```bash
cosmos-curate local launch \
    --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
    -- pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
    --input-video-path <input path> \
    --output-clip-path <output path> \
    --enhance-captions \
    --enhance-captions-lm-variant openai \
    --enhance-captions-openai-model gpt-5.1-20251113 \
    --enhance-captions-max-output-tokens 2048
```

`--enhance-captions-openai-model` selects the OpenAI API model (default `gpt-5.1-20251113`). Set `openai.base_url` in config if you need to use a custom base URL. You can increase `--enhance-captions-max-output-tokens` if you need longer rewrites; the default `2048` works for most scenarios.

4. **Optionally, Run the Split-Annotate Pipeline via API Endpoint**

First, launch the container with a service endpoint.

```bash
cosmos-curate local launch \
   --image-name cosmos-curate --image-tag 1.0.0 --curator-path . \
   -- pixi run python cosmos_curate/scripts/onto_nvcf.py --helm False
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

To stop the service, press `Ctrl+C` in the terminal where you ran the launch command.

**Note:** When Ray starts up, it will print instructions including `ray stop` - do not use this command directly on
the host. The Ray cluster runs inside the container and is managed by the service. Always stop the service using
`Ctrl+C` for proper cleanup.

### Generate Dataset for Cosmos-Predict2 Post-Training

The [Split-Annotate Pipeline](../curator/REFERENCE_PIPELINES_VIDEO.md#split-annotate-pipeline) above has first-class support
for [Cosmos-Predict2 Video2World post-training](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world.md).

The following arguments are needed for `split-annotate` pipeline to generate the datasets for Cosmos-Predict2:
- add `--generate-cosmos-predict-dataset predict2` to enable the dataset creation.
- add ` --transnetv2-min-length-frames 120` to specify a minimum clip length of (e.g.) 120 frames, as Cosmos-Predict2 post-training requires 93 frames by default.

This will generate a `cosmos_predict2_video2world_dataset/` sub-directory under the output path specified by `output_clip_path`.
The `cosmos_predict2_video2world_dataset/` sub-directory has the following structure:

```bash
cosmos_predict2_video2world_dataset/
├── metas/
│   ├── {clip-uuid}_{start_frame}_{end_frame}.txt
├── videos/
│   ├── {clip-uuid}_{start_frame}_{end_frame}.mp4
├── t5_xxl/
│   ├── {clip-uuid}_{start_frame}_{end_frame}.pickle
```

Note the `T5` embedding generation are included in this pipeline, such that there is no need to
run the `python -m scripts.get_t5_embeddings --dataset_path ...` command (from `Cosmos-Predict2` repo) as stated in the
[post-training guide](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world.md#post-training-guide),
unless you want to manually edit the captions which will require re-generating the T5 embeddings.

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

One trick to make things easier is to define `my-slurm-login-01.my-cluster.com` in your `~/.ssh/config`, like

```bash
Host my-slurm-login-01.my-cluster.com
  HostName <my-real-slurm-cluster-hostname>
```

Then you can login to your cluster literally using `ssh my-slurm-login-01.my-cluster.com`.

#### Identify User Path on the Cluster

Assume your user directory on the Slurm cluster is `/home/myusername/`. Note
- this path should be **accessible to all compute nodes**.
- this path should have enough disk quota to hold the image and model weights.

Set `$SLURM_USER_DIR` environment variable **on your local host**.

```bash
export SLURM_USER_DIR="/home/myusername"
```

Then set other dependent environment variables.

```bash
source examples/slurm/source_me_env_vars.sh
```

These helper scripts detect the presence of AWS and Azure credential files and only mount the ones that exist, so clusters that rely on a single object store do not need to stage the other provider's configuration.

### Copy Config File, Cloud Storage Credentials, and Model Files to Cluster

**Note**: Cloud storage credentials are only required if your input videos or output paths use S3/Azure URIs. If all data resides on the cluster's local or shared storage, you can skip the credential sync steps below.

If you have defined `my-slurm-login-01.my-cluster.com` in your `/.ssh/config` like mentioned above, you can simply run

```bash
./examples/slurm/sync_config_creds_models.sh
```

Otherwise you can replace `my-slurm-login-01.my-cluster.com` with your real login hostname in the following commands.

```bash
# Copy ~/.config/cosmos_curate/config.yaml
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_COSMOS_CURATE_CONFIG_DIR}
rsync -avh ~/.config/cosmos_curate/config.yaml my-slurm-login-01.my-cluster.com:${SLURM_COSMOS_CURATE_CONFIG_DIR}/config.yaml

# (Optional) Copy ~/.aws/credentials if using S3-compatible storage
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_AWS_CREDS_DIR}
rsync -avh ~/.aws/credentials my-slurm-login-01.my-cluster.com:${SLURM_AWS_CREDS_DIR}/credentials

# (Optional) Copy ~/.azure/credentials if using Azure Blob Storage
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_AZURE_CREDS_DIR}
rsync -avh ~/.azure/credentials my-slurm-login-01.my-cluster.com:${SLURM_AZURE_CREDS_DIR}/credentials

# Copy models
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_WORKSPACE}/models
rsync -avh --progress ${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace/models/  my-slurm-login-01.my-cluster.com:${SLURM_WORKSPACE}/models/
```

### Create sqsh Image and Copy to the Slurm Cluster

1. Install `enroot` on your local machine based on the [instructions here](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md).

2. Import the hello world docker image built above to create a `.sqsh` file.

```bash
export COSMOS_CURATE_IMAGE_NAME="cosmos-curate_hello-world.sqsh"
enroot import --output $COSMOS_CURATE_IMAGE_NAME dockerd://cosmos-curate:hello-world
```

3. Copy the sqsh file to slurm cluster.

Again if you have defined `my-slurm-login-01.my-cluster.com` in your `/.ssh/config`, you can simply run

```bash
./examples/slurm/upload_image.sh
```

Otherwise replace `my-slurm-login-01.my-cluster.com` with your real login hostname in the following commands.

```bash
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_IMAGE_DIR}
rsync -avh --progress $COSMOS_CURATE_IMAGE_NAME "my-slurm-login-01.my-cluster.com:${SLURM_IMAGE_DIR}/"
```

### Launch on Slurm

Figure out the following information for your Slurm cluster
- your user name on the slurm cluster, in case it is different than your local machine
- slurm account you are in
- slurm partition to use
- GRES, e.g. `gpu:4` or `gpu:8`

Launch!

```bash
cosmos-curate slurm submit \
  --login-node my-slurm-login-01.my-cluster.com \
  --username my_username_on_slurm_cluster_if_different_than_local_username \
  --account my_slurm_account \
  --partition my_slurm_gpu_partition \
  --gres=my_slurm_cluster_gres \
  --num-nodes 1 \
  --job-name "hello-world" \
  --remote-files-path "${SLURM_USER_DIR}/job_info" \
  --container-image "${SLURM_IMAGE_DIR}/${COSMOS_CURATE_IMAGE_NAME}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    -- python -m cosmos_curate.pipelines.examples.hello_world_pipeline
```

The command above will print the slurm job id like below
```bash
Submitted batch job <slurm_job_id>
```

#### Email Notifications (Optional)

You can optionally receive email notifications about your SLURM jobs using the `--mail-user` and `--mail-type` parameters.

**Parameters:**
- `--mail-user`: Email address to receive notifications
- `--mail-type`: Comma-separated list of events to notify about. Options include:
  - `BEGIN`: Job start
  - `END`: Job completion
  - `FAIL`: Job failure
  - `REQUEUE`: Job requeue
  - `ALL`: All events
  - `STAGE_OUT`: Stage out (data transfer) completion
  - `TIME_LIMIT`, `TIME_LIMIT_90`, `TIME_LIMIT_80`: Time limit warnings

**Note:** If you provide `--mail-user` without `--mail-type`, SLURM will typically default to `END,FAIL`. If you provide `--mail-type`, you must also provide `--mail-user`.

**Example with email notifications:**

```bash
cosmos-curate slurm submit \
  --login-node my-slurm-login-01.my-cluster.com \
  --username my_username_on_slurm_cluster_if_different_than_local_username \
  --account my_slurm_account \
  --partition my_slurm_gpu_partition \
  --gres=my_slurm_cluster_gres \
  --num-nodes 1 \
  --job-name "hello-world" \
  --remote-files-path "${SLURM_USER_DIR}/job_info" \
  --container-image "${SLURM_IMAGE_DIR}/${COSMOS_CURATE_IMAGE_NAME}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
  --mail-user your.email@example.com \
  --mail-type END,FAIL \
    -- python -m cosmos_curate.pipelines.examples.hello_world_pipeline
```

**⚠️ Cluster-Specific Considerations:**

Email notification functionality depends on your cluster's configuration:
- The cluster must have a properly configured mail server
- Network firewalls or security policies may affect email delivery
- Some clusters may have email notifications disabled or restricted by policy
- Email delivery may be delayed depending on the mail server configuration

If email notifications are not working as expected, please verify with your cluster administrators that this feature is enabled and properly configured for your environment.

### Find Logs

The slurm job log is at `"${SLURM_LOG_DIR}/{job_name}_{slurm_job_id}.log"` on the cluster.

You can also use the CLI to monitor the log:

```bash
cosmos-curate slurm job-log \
  --login-node my-slurm-login-01.my-cluster.com \
  --username my_username_on_slurm_cluster_if_different_than_local_username \
  --job-id slurm_job_id_printed_above
```

### Developing on Slurm

If you plan to modify or create new pipelines on slurm, it is useful to mount the source code into the container so that you do not need to rebuild the container for every change.

#### Add the source-code mount to the `$CONTAINER_MOUNTS`:

This will use the source code that is located at `"${SLURM_SOURCE_DIR}/cosmos_curate/"` on the cluster and will override the source that is inside the container.

```bash
source examples/slurm/source_me_source_code_mount.sh
```

#### Sync source code to the slurm cluster

Again if you have defined `my-slurm-login-01.my-cluster.com` in your `/.ssh/config`, you can simply run

```bash
./examples/slurm/sync_source_code.sh
```

Otherwise replace `my-slurm-login-01.my-cluster.com` with your real login hostname in the following commands.

```bash
ssh my-slurm-login-01.my-cluster.com mkdir -p ${SLURM_SOURCE_DIR}/cosmos_curate/
rsync -avh ./cosmos_curate/ my-slurm-login-01.my-cluster.com:${SLURM_SOURCE_DIR}/cosmos_curate/
```

Note that:

```bash
cosmos-curate slurm submit ...
```

will not sync code from your local machine to the cluster. You'll need to either edit code directly on the cluster, or call the above source-code sync-ing command(s) again.

### Speeding up Model Load on Slurm

Model loading can sometimes be sped up by adding

```
--copy-weight-to /raid/scratch/models
```

to the launch command. This is because of the way that the transformers library loads safetensors from disk. Its method is efficient for local disk, but can be very slow if the model weights are stored on NFS or Lustre.

This is only useful if there is enough space on the local nodes to store the model weights.

## Launch Pipelines on NVIDIA DGX Cloud

Cosmos-Curate can be deployed on [NVIDIA Cloud Function (NVCF)](https://docs.nvidia.com/cloud-functions/index.html) platform.

There are a few steps needed to get a new user onboarded to NVCF, so please reach out to NVIDIA Cosmos-Curate team and we will guide you through the process.

If you have already onboarded to NVCF and have an NVCF Org, please follow this [NVCF Guide](NVCF_GUIDE.md) to deploy Cosmos-Curate on NVCF.

## Launch Pipelines on K8s Cluster (coming soon)

## Observability for Pipelines

The resource usage and bottleneck of the pipeline can vary with:
- input data, e.g. when you have ~10MB videos vs. ~10GB videos,
  or in the most difficult case you have a mix of 10MB & 10GB videos in the same input set;
- hardware configuration, e.g. ratio of CPU core count & GPU count & system memory size.

Therefore, it is critical to have good observability in place to help debug reliability problems and optimize pipeline throughput.

We have implemented a set of metrics in [Cosmos-Xenna](https://github.com/nvidia-cosmos/cosmos-xenna)
and included a [Grafana dashboard](../../examples/observability/grafana/cosmos-curate-oss.json) for `Cosmos-Curate` pipelines.
More details can be found in [Observability Guide](../curator/OBSERVABILITY_GUIDE.md).

## Build the Client package

The `cosmos-curate` client can be built as a wheel and installed in a standalone mode, without the need for the rest of the source environment

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
