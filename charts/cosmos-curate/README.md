# Cosmos-Curate Helm Chart

## Pre-requisites
* [Helm binary](https://helm.sh/docs/intro/install/)
* [OpenTelemetry](https://opentelemetry.io/) repo
* [NGC CLI and NGC profile](https://org.ngc.nvidia.com/setup/installers/cli) (if pushing a new chart to the [NGC registry](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html))
* `NGC_NVCF_ORG` - NVCF org ID, i.e. destination repo for chart to be pushed

## One-Time Steps

### Add Open-Telemetry repo
```bash
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo update
```

### Config NGC CLI
```bash
ngc config set
# fill in
# - your API key
# - "ascii" for output format type
# - your NVCF ORG ID
# - "no-team" for the team unless you are explicitly told to use certain NVCF team
# - "no-ace"
```

### Create the Chart Metadata for New Orgs:
```bash
ngc registry chart create --short-desc "Chart for NVCF function for cosmos curate" ${NGC_NVCF_ORG}/cosmos-curate
```

## Version Control
The latest version right now is `2.1.1`; you can edit this version as needed if you are tweaking this chart.
```bash
export CHART_VERSION=2.1.1
```

## Manually Packaging a chart
```bash
helm dep build charts/cosmos-curate/
helm package charts/cosmos-curate --version ${CHART_VERSION}
```

## Publishing to NGC helm registry
```bash
# Push the .tgz
ngc registry chart push ${NGC_NVCF_ORG}/cosmos-curate:${CHART_VERSION}
```

## Removing a chart (if you REALLY need to replace an existing version, use with caution) 
```bash
ngc registry chart remove ${NGC_NVCF_ORG}/cosmos-curate:${CHART_VERSION}
```
