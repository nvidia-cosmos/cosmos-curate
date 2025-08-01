# cosmos-curate Helm
## Pre-requisites
* [helm binary](https://helm.sh/docs/intro/install/)
* [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) and NGC profile (if pushing a new chart to the registry)
* NGC_NVCF_API_KEY env var (if calling create/deploy) - that is an API key for an ORG with NVCF access
* NGC_NVCF_ORG - destination repo for chart to be pushed

## One-time steps

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

## Create the Chart Metadata for New Orgs:
```bash
ngc registry chart create --short-desc "Chart for NVCF function for cosmos curate" ${NGC_NVCF_ORG}/cosmos-curate
```

## Manually Packaging a chart
```bash
export CHART_VERSION=2.0.5
helm dep build charts/cosmos-curate/
helm package charts/cosmos-curate --version ${CHART_VERSION}
```

## Publishing to NGC helm registry
```bash
# Push the .tgz
ngc registry chart push ${NGC_NVCF_ORG}/cosmos-curate:${CHART_VERSION}
```

## Removing a chart (optional - use if you need to replace an existing version, use with caution) 
```bash
ngc registry chart remove ${NGC_NVCF_ORG}/cosmos-curate:${CHART_VERSION}
```
