# Cosmos Curator Next

> “Be water, my friend.” — Bruce Lee

## Vision

> Cosmos Curator is an agent-friendly toolkit for building physical AI data curation workflows.

Physical AI curation varies widely across teams and use cases. The field is still evolving, and there is no universal
end-to-end pipeline that Curator should encode as the product.

Curator therefore treats pipelines as compositions, not end products. People and coding agents use its reusable building
blocks and reference recipes to create their own workflows.

Curator's durable value is the domain knowledge that connects execution, storage, and inference. It provides useful
representations and operations for physical AI data, together with ways to inspect their results.

Fast iteration is a first-class goal. Components can run independently and on small samples, so users can inspect results
and revise workflows before scaling. Repeated dataset refinement matters as much as peak throughput.

At scale, Curator aspires to the speed of light. Components and recipes should expose bottlenecks and tune Ray Data
workflows toward the practical limits of the underlying GPU cluster.

## Product Boundary

- **Ray Data** is the primary execution and composition layer.
- **Lance** is the default durable format for tabular datasets; media and bulk sensor payloads remain in user-selected
  storage and are referenced from the dataset.
- **vLLM** is the preferred backend for high-throughput LLM inference, with **Transformers** providing upstream model and
  processor support. Curator adds model-specific code only for curation needs. Other inference systems can integrate
  through adapters.
- **Cosmos Curator** provides reusable physical AI building blocks and reference recipes.
- **User code** makes the curation decisions.

These systems remain visible in user code and operations. Curator connects them for physical AI curation while
preserving their native interfaces.

## Intended Workflow

Conceptually, a Curator workflow looks like this:

```text
       media/sensor payload references + metadata
                            |
                            v
+------------------------------------------------------+
| user-authored Ray Data program                       |
|                                                      |
| - combines Curator components with user functions    |
| - calls model backends through Curator adapters      |
| - applies user policy and validates results          |
+------------------------------------------------------+
                            |
                            v
       derived Lance dataset + payload references
```

Tabular sensor data can live directly in Lance. Payload references point to media and other bulk sensor data in
user-selected storage.

The center box lists capabilities available inside the program, not required stages or a class hierarchy. Users choose
the operations and ordering their workflow requires.

## Reusable Toolkit

Curator focuses on a few reusable areas:

- **Media and sensors.** Prepare source data through discovery, decoding, sampling, and sensor alignment.
- **Models and selection.** Generate annotations, embeddings, and scores; support clustering, deduplication, and
  selection.
- **Validation and integration.** Define shared schemas and provenance, make results inspectable, and connect them to
  downstream tools.

Shared schemas give components a common vocabulary while allowing customer-specific extensions. Their exact
representation can evolve as the toolkit gains implementation experience.

## Agent-Friendly Workflows

Agent-friendly has two complementary meanings:

- **Authoring.** Components expose clear, typed interfaces that coding agents can discover and combine.
- **Operations.** Structured inputs and machine-readable results make runs reproducible and diagnosable by agentic tools.

Two design notes explore the operations layer from different ends: the [Orca design note](orca.md) covers agentic
orchestration, and the [managed Ray clusters design](curator-next-slurm-ray.md) covers the cluster substrate those
runs execute on. Both remain separate from the reusable toolkit; Curator components and recipes remain usable without
either.

## Reference Recipes

At a logical level, Curator Next keeps the reusable toolkit and reference recipes separate:

```text
Curator Next
|-- reusable toolkit
|   |-- components and shared schemas
|   `-- model integrations and validation
`-- reference recipes
    |-- workflow composition and configuration
    `-- recipe-specific artifacts
```

The labels are conceptual rather than prescribed package names. Recipe-specific code stays with its recipe; behavior
and schemas that are useful across workflows belong in the shared toolkit. Recipes are maintained starting points, not
the primary product API. When reuse is uncertain, code stays with a recipe until another workflow needs it.

Regular runs and benchmarks verify that the components still work well together. Performance changes are justified by
recipe benchmarks.

## Incubation

`cosmos_curator.next` is the temporary incubation namespace for the Curator Next programming model. It isolates new work
during the transition; it is not a commitment to the long-term public package layout or stable import paths.

New Curator Next components and reference recipes develop here. Existing code, including current Ray Data work, moves
here only through explicit transition work. Code and tests define current behavior during incubation.

Once components are proven across multiple workflows, they graduate into namespaces named for durable responsibilities.
Graduation includes an explicit transition plan for code and users.

The Xenna stage model is not the Curator Next programming model. Existing pipelines remain available during the
transition.

## Non-Goals

Curator Next does not:

- replace Ray Data with its own runtime or pipeline language
- become a general orchestrator or model-serving platform
- own customer infrastructure or training formats
- recreate every feature of the current pipelines
