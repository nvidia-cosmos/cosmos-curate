# Cosmos-Curate - Pipeline Design Guide

- [Cosmos-Curate - Pipeline Design Guide](#cosmos-curate---pipeline-design-guide)
  - [Core Components](#core-components)
    - [Pipeline Task Class](#pipeline-task-class)
    - [Pipeline Stage Class](#pipeline-stage-class)
    - [Model Class](#model-class)
    - [Conda Environment Management](#conda-environment-management)
    - [Run the Pipeline](#run-the-pipeline)
      - [StageSpec Customization](#stagespec-customization)
  - [Quiz: Adding WordCount Stage to Hello-World Pipeline](#quiz-adding-wordcount-stage-to-hello-world-pipeline)
  - [Pipeline Performance](#pipeline-performance)
    - [Performance Monitoring](#performance-monitoring)
    - [Improving GPU Utilization](#improving-gpu-utilization)
      - [Extract CPU Processing to Separate Stages](#extract-cpu-processing-to-separate-stages)
      - [Pack Multiple GPU Workers to One GPU](#pack-multiple-gpu-workers-to-one-gpu)

This guide explains how to modify existing pipelines or add new pipelines into the Cosmos-Curate system.

## Core Components

Let's use the [hello_world_pipeline](../../cosmos_curate/pipelines/examples/hello_world_pipeline.py) as an example to walk through the core components.

### Pipeline Task Class

Each new pipeline should define a class representing the tasks being passed between stages.

This class need to inherit `PipelineTask` base class defined in
[cosmos_curate/core/interfaces/stage_interface.py](../../cosmos_curate/core/interfaces/stage_interface.py).
No function overriding is needed today but may needed later as the underlying layer improves.

For the Hello-World pipeline, it defines a simple `HelloWorldTask` class:

```python
@attrs.define
class HelloWorldTask(PipelineTask):
    prompt: str
    output: str | None = None
```

What typically happens is
- first construct a list of input tasks with a few attributes initialized
- as the task passing through pipeline stages, more fields are getting populated.

For example, it builds a list of two input tasks
```python
prompts = ["The KEY TO A CREATING GOOD art is", "Once upon a time"]
tasks: list[PipelineTask] = [HelloWorldTask(prompt=x) for x in prompts]
```

### Pipeline Stage Class

Each pipeline stage should define a class that inherits `CuratorStage` base class defined in
[cosmos_curate/core/interfaces/stage_interface.py](../../cosmos_curate/core/interfaces/stage_interface.py).

**The following methods need to be overridden always:**

1. `process_data()`: implements the actual actions for this stage.
   - The method takes a list of pipeline tasks as input and output a list of pipeline tasks.
   - In the hello-world pipeline, the `_LowerCaseStage` simply convert the `prompt` field to lower case in each pipeline task.
   - There are two more advanced use cases:
     - The number of input tasks can be different than the number of output tasks.
       - This is the "dynamic chunking" feature discussed in the [How to handle large variation in input data?](./ARCHITECTURE_GUIDE.md#how-to-handle-large-variation-in-input-data) section.
       - The other [demo_task_chunking_pipeline](../../cosmos_curate/pipelines/examples/demo_task_chunking_pipeline.py) demonstrates the feature with a similar minimal example.
     - The type of input tasks can be different than the type of output tasks.

```python
class _LowerCaseStage(CuratorStage):

    def process_data(self, tasks: list[HelloWorldTask]) -> list[HelloWorldTask] | None:
        # convert the prompt to lowercase
        for task in tasks:
            task.prompt = task.prompt.lower()
        return tasks
```

2. `resources()`: specifies how many CPUs and GPUs each stage worker need
   - The `cpus` represents the number of logical CPU cores that the stage worker will use.
   - The `gpus` can be either
     - a fractional number like `0.25` if this stage's worker cannot fully utilize one GPU.
     - a `>1` number to allocate more than one GPU to enable e.g. tensor-parallelism for running larger models.

```python
    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=1.0, gpus=0.0)
```

**For a stage that uses an AI model, the following methods need to be overriden:**

1. `model()`: returns a `ModelInterface` object.
   - This is needed by the underlying layer to derive pipeline runtime configuration.

```python
class _GPT2Stage(CuratorStage):

    @property
    def model(self) -> ModelInterface | None:
        return self._model
```

2. `stage_setup()`: implements the setup work when a Ray actor for this stage's worker is created.
   - A subtle but important difference between the `__init__` constructor and this `stage_setup` method is that the constructor runs in the base conda environment while `stage_setup` runs in the conda environment specified by the model. Therefore code to setup the model need to be in this method.
   - The CPU stages without a model can also implement this method if any setup work is needed.

```python
    def stage_setup(self) -> None:
        self._model.setup()
```

### Model Class

Each model should be wrapped in a class, which inherits `ModelInterface` base class defined in
[cosmos_curate/core/interfaces/model_interface.py](../../cosmos_curate/core/interfaces/model_interface.py).

The hello-world pipeline uses `GPT2` model defined in [cosmos_curate/models/gpt2.py](../../cosmos_curate/models/gpt2.py).

The following methods are required to be overriden:

```python
class GPT2(ModelInterface):
    # need override conda_env_name to tell underlying logic which conda env to use
    @property
    def conda_env_name(self) -> str:
        return "transformers"

    # need override model_id_names to faciliate model download
    @property
    def model_id_names(self) -> list[str]:
        return ["openai-community/gpt2"]

    # need override setup which is called automatically when creating stage actors
    def setup(self) -> None:
        model_dir = model_utils.get_local_dir_for_weights_name(self.model_id_names[0])
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to("cuda")
```

To help management of models, add a section in [all_models.json](../../cosmos_curate/configs/all_models.json).
- If only a few files are needed from the huggingface repo, the list of file names can be specified under `filelist` entry.
- When running on [NVIDIA Cloud Function](../client/NVCF_GUIDE.md#upload-model-weights), the model ID used on NVCF model registry can be specified under `nvcf_model_id` entry.

```json
{
    ...,
    "gpt2": {
        "model_id": "openai-community/gpt2",
        "version": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "filelist": null,
        "nvcf_model_id": "gpt2"
    },
    ...
}
```

### Conda Environment Management

The `GPT2` model above uses a conda environment called `transformers`;
that corresponds to directory `transformers/` under [package/cosmos_curate/envs/](../../package/cosmos_curate/envs/).

For every "env" there, a single `jinja2` templated dockerfile snippet named `build_steps.dockerfile.j2` is required
to have the recipes for building this conda environment. For example,

```dockerfile
# /transformers/build_steps.dockerfile.j2
RUN {{cache_mount_str}} micromamba install python=3.10.14 -y
RUN {{cache_mount_str}} pip install torch==2.3.1 transformers==4.41.2

{{install_regular_cosmos_curator_deps_str}}
```

Then when building the docker image for running pipelines, use option `--envs` to specify which conda environments to be included in the build.

### Run the Pipeline

Once we have
- a list of input pipeline tasks
- a list of pipeline stages

We can call `run_pipeline(input_tasks: list[PipelineTaks], stages: list[CuratorStage | CuratorStageSpec])` defined in [cosmos_curate/core/interfaces/pipeline_interface.py](../../cosmos_curate/core/interfaces/pipeline_interface.py).

In hello-world pipeline, 

```python
    # construct a list of input pipeline tasks
    prompts = ["The KEY TO A CREATING GOOD art is", "Once upon a time"]
    tasks: list[PipelineTask] = [HelloWorldTask(prompt=x) for x in prompts]

    # construct a list of pipeline stages
    stages: list[CuratorStage | CuratorStageSpec] = [
        CuratorStageSpec(_LowerCaseStage(), num_workers_per_node=2),
        _PrintStage(),
        _GPT2Stage(),
    ]

    # run the pipeline
    run_pipeline(tasks, stages)
```

#### StageSpec Customization

When building the pipeline, each `CuratorStage` class is wrapped by a `CuratorStageSpec` class with a list of properties.
To override these properties, we can wrap the stage class directly before sending to `run_pipeline`.
Two commonly used properties are:
- `num_workers_per_node: int = None`: set a fixed number of workers per node and disable auto-scaling, typically for IO workers.
- `num_run_attempts_python: int = 1`: set number of retry attempts, if randomly failures are expected.

## Quiz: Adding WordCount Stage to Hello-World Pipeline

As a simple exercise, consider adding a `WordCountStage` after the `_GPT2Stage` to count the number words that GPT2 has generated.
- Add a field `word_count` to `HelloWorldTask(PipelineTask)`
- Define a new `WordCountStage(CuratorStage)` and implement `resources()`, `process_data`, etc.
- Add the stage to `stages: list[CuratorStage | CuratorStageSpec]` before sending to `run_pipeline`.
- Run the new pipeline with instructions in [End User Guide](../client/END_USER_GUIDE.md#run-the-hello-world-example-pipeline).

## Pipeline Performance

### Performance Monitoring

[Prometheus](https://prometheus.io/)-compatible metrics are exported at port `localhost:9002/metrics`.

A list of useful [PromQL](https://prometheus.io/docs/prometheus/latest/querying/basics/) queries for performance debugging are summarized below:

```bash
# Measured stage speed, i.e. process time per task per actor for each stage
sum by (stage) (ray_pipeline_actor_process_time)

# Number of busy vs. idle workers per stage
sum by (stage, state) (ray_pipeline_actor_count{state!="target", state!="pending"})

# Input & output queue sizes per stage
sum by (stage) (ray_pipeline_input_queue_size)
sum by (stage) (ray_pipeline_output_queue_size)

# Cross-stage object size
(sum by (stage) (ray_pipeline_stage_deserialize_size_total))
/ on (stage) group_left ()
(sum by (stage) (ray_pipeline_stage_deserialize_count_total))

# Communication time / process time; i.e. are we able to hide cross-stage data movement
(sum by (stage) (ray_pipeline_stage_deserialize_time_total))
/ on (stage) group_left ()
(sum by (stage) (ray_pipeline_stage_process_time_total))

# GPU utilization averaged by stage
avg by (stage) (
    ray_pipeline_stage_gpu_alloc * on (SessionName, NodeAddress, GpuIndex) group_left
    label_replace(ray_node_gpus_utilization, "NodeAddress","$1","ip", "(.+)")
)

# GPU memory usage averaged by stage
avg by (stage) (
    ray_pipeline_stage_gpu_alloc * on (SessionName, NodeAddress, GpuIndex) group_left
    label_replace(ray_node_gram_used, "NodeAddress","$1","ip", "(.+)")
)

# CPU usage aggregated per stage
sum by (stage) (ray_pipeline_actor_resource_usage{stage!="", resource="cpu"}) / 100

# Average CPU usage per actor for each stage
(sum by (stage) (ray_pipeline_actor_resource_usage{stage!="", resource="cpu"}))
/ on (stage)
(sum by (stage) (ray_pipeline_actor_count{state="running"})) / 100

# System memory usage aggregated per stage
sum by (stage) (ray_pipeline_actor_resource_usage{stage!="", resource="memory"})
```

And a sample [Grafana](https://grafana.com/) dashboard will be released soon.

### Improving GPU Utilization

GPU utilization is a first-order metric to gauge the pipeline performance.
Down to the model inference, there are many optimization techniques;
but in this section, we will focus on the pipeline-level considerations to improve GPU utilization.

#### Extract CPU Processing to Separate Stages

Some models require non-trivial preprocessing on CPU before sending to GPU for inference.
For example, vision models would require the input video to be first decoded into frames and then resize, normalized, etc.
If such preprocessing happens inside the GPU stage, it will waste GPU time.

This framework makes it very easy to address such problems.
We can simply extract such CPU work into a separate stage and pass the processed tensors into the GPU stage directly.
As a result, the GPU stage will push the GPU utilization higher while the CPU stage would be effectively free by hidding under the GPU processing time.
The `QwenInputPreparationStage` and `QwenCaptionStage` in [cosmos_curate/pipelines/video/captioning/captioning_stage.py](../../cosmos_curate/pipelines/video/captioning/captioning_stages.py) are an example of such optimizations.

#### Pack Multiple GPU Workers to One GPU

For small models, sometimes it is difficult to push up the GPU utilization.
One way is to request a fraction of GPU and allow multiple stage workers to be allocated on the same GPU.
The GPU memory usage metric above would help define this fractional number to maximize the GPU usage while avoiding CUDA OOM.

In the reference pipeline, `AestheticFilterStage` in [cosmos_curate/pipelines/video/filtering/aesthetics/aesthetic_filter_stages.py](../../cosmos_curate/pipelines/video/filtering/aesthetics/aesthetic_filter_stages.py) requests 0.25 GPUs per worker.

