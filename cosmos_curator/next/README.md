# Curator Next

`cosmos_curator.next` is the temporary incubation namespace for the Cosmos Curator Next programming model.

New Curator Next components and reference recipes develop here without committing to their eventual public package
layout. Existing code, including current Ray Data work, moves here only through explicit transition work. The underlying
execution, storage, and model ecosystems remain visible rather than being hidden behind a Curator-specific runtime.

The package layout and import paths are provisional. Once components are proven across multiple workflows, they graduate
into namespaces named for durable responsibilities through an explicit transition plan.

Existing pipelines remain available during the transition. The Xenna stage model is not the Curator Next programming
model.

Coding agents and contributors starting a new recipe should first read the
[Curator Next agent guidelines](AGENTS.md). They capture the current recipe-first working agreement, implementation
constraints, and managed-Ray qualification criteria while the longer-term design remains in flux.

See [Cosmos Curator Next](../../docs/curator/design/curator-next.md) for the design direction.
