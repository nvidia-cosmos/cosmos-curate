{{/*
Expand the name of the chart.
*/}}
{{- define "curator-ray.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "curator-ray.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "curator-ray.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "curator-ray.labels" -}}
helm.sh/chart: {{ include "curator-ray.chart" . }}
{{ include "curator-ray.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "curator-ray.selectorLabels" -}}
app.kubernetes.io/name: {{ include "curator-ray.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Source package root inside the curator image. This is baked into the image, so it
is intentionally not a values.yaml knob: it must stay in sync with the dockerfile
WORKDIR/ENV PYTHONPATH and CONTAINER_PATHS_CODE_DIR (cosmos_curator/core/utils/environment.py).
*/}}
{{- define "curator-ray.codeDir" -}}
/opt/cosmos-curator
{{- end }}

{{/*
Effective shared OTLP transport values.

Top-level `otlp.*` is the shared OTLP transport for logs, traces, and direct
in-process metrics push. `metrics.otlp.*` is a metrics-specific transport
override for the chart-managed collector's OTLP metrics exporter.
*/}}
{{- define "curator-ray.otlp.endpoint" -}}
{{- $metricsEndpoint := .Values.metrics.otlp.endpoint | toString -}}
{{- if contains "${env:" $metricsEndpoint -}}
{{- $metricsEndpoint = "" -}}
{{- end -}}
{{- or .Values.otlp.endpoint $metricsEndpoint -}}
{{- end }}

{{- define "curator-ray.otlp.usesLegacyMetricsEndpoint" -}}
{{- $metricsEndpoint := .Values.metrics.otlp.endpoint | toString -}}
{{- if and (not .Values.otlp.endpoint) $metricsEndpoint (not (contains "${env:" $metricsEndpoint)) -}}true{{- end -}}
{{- end }}

{{- define "curator-ray.otlp.certPath" -}}
{{- or .Values.otlp.tls.certPath "/etc/curator-otlp/certs/tls.crt" -}}
{{- end }}

{{- define "curator-ray.otlp.keyPath" -}}
{{- or .Values.otlp.tls.keyPath "/etc/curator-otlp/certs/tls.key" -}}
{{- end }}

{{- define "curator-ray.otlp.caPath" -}}
{{- default "" .Values.otlp.tls.caPath -}}
{{- end }}

{{/*
Certificate verification for pod-local OTLP consumers: the log sidecar and any
in-process exporter. Only `otlp.tls.insecureSkipVerify` relaxes it. The legacy
`metrics.otlp.insecureSkipVerify` stays scoped to the chart-managed metrics
collector, the same way its certPath/keyPath do, so an exception made for one
backend cannot silently disable verification against another.
*/}}
{{- define "curator-ray.otlp.insecureSkipVerify" -}}
{{- if kindIs "bool" .Values.otlp.tls.insecureSkipVerify -}}
{{- .Values.otlp.tls.insecureSkipVerify -}}
{{- else -}}
false
{{- end -}}
{{- end }}

{{/*
Certificate verification for the chart-managed metrics collector's OTLP
exporter. Shared `otlp.tls.insecureSkipVerify` wins when set; otherwise the
legacy metrics-scoped value applies, preserving 2.3 behavior.
*/}}
{{- define "curator-ray.metricsOtlp.insecureSkipVerify" -}}
{{- if kindIs "bool" .Values.otlp.tls.insecureSkipVerify -}}
{{- .Values.otlp.tls.insecureSkipVerify -}}
{{- else -}}
{{- .Values.metrics.otlp.insecureSkipVerify | default false -}}
{{- end -}}
{{- end }}

{{- define "curator-ray.otlp.runAttributes" -}}
{{- $attrs := deepCopy (default (dict) .Values.metrics.extraExternalLabels) -}}
{{- $attrs = merge $attrs (default (dict) .Values.metrics.externalLabels) -}}
{{- $attrs | toJson -}}
{{- end }}

{{- define "curator-ray.otlp.enabled" -}}
{{- if or .Values.logging.otlp.enabled .Values.metrics.otlpPush.enabled .Values.tracing.otlp.enabled -}}true{{- end -}}
{{- end }}

{{- define "curator-ray.otlp.appEnv" -}}
{{- $customEnv := default (dict) .Values.customEnvVars -}}
{{- $appOtlpEnabled := or .Values.metrics.otlpPush.enabled .Values.tracing.otlp.enabled -}}
{{- $otlpEndpoint := include "curator-ray.otlp.endpoint" . -}}
{{- $otlpCertPath := include "curator-ray.otlp.certPath" . -}}
{{- $otlpKeyPath := include "curator-ray.otlp.keyPath" . -}}
{{- $otlpCaPath := include "curator-ray.otlp.caPath" . -}}
{{- $otlpCertsEnabled := or .Values.otlp.tls.secret.enabled .Values.otlp.extractNVCFSecrets -}}
{{- $explicitOtlpClientCerts := and .Values.otlp.tls.certPath .Values.otlp.tls.keyPath -}}
{{- $otlpRunAttributes := include "curator-ray.otlp.runAttributes" . | fromJson -}}
{{- if and $appOtlpEnabled (not (hasKey $customEnv "OTEL_EXPORTER_OTLP_ENDPOINT")) }}
OTEL_EXPORTER_OTLP_ENDPOINT: {{ $otlpEndpoint | quote }}
{{- end }}
{{- if and $appOtlpEnabled (not (hasKey $customEnv "OTEL_EXPORTER_OTLP_PROTOCOL")) }}
OTEL_EXPORTER_OTLP_PROTOCOL: {{ .Values.otlp.protocol | quote }}
{{- end }}
{{- if and $appOtlpEnabled .Values.otlp.timeout (not (hasKey $customEnv "OTEL_EXPORTER_OTLP_TIMEOUT")) }}
OTEL_EXPORTER_OTLP_TIMEOUT: {{ .Values.otlp.timeout | quote }}
{{- end }}
{{- if and $appOtlpEnabled (or $otlpCertsEnabled $explicitOtlpClientCerts) $otlpCertPath (not (hasKey $customEnv "OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE")) }}
OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE: {{ $otlpCertPath | quote }}
{{- end }}
{{- if and $appOtlpEnabled (or $otlpCertsEnabled $explicitOtlpClientCerts) $otlpKeyPath (not (hasKey $customEnv "OTEL_EXPORTER_OTLP_CLIENT_KEY")) }}
OTEL_EXPORTER_OTLP_CLIENT_KEY: {{ $otlpKeyPath | quote }}
{{- end }}
{{- if and $appOtlpEnabled $otlpCaPath (not (hasKey $customEnv "OTEL_EXPORTER_OTLP_CERTIFICATE")) }}
OTEL_EXPORTER_OTLP_CERTIFICATE: {{ $otlpCaPath | quote }}
{{- end }}
{{- if and .Values.metrics.otlpPush.enabled (not (hasKey $customEnv "COSMOS_CURATOR_OTLP_METRICS_PUSH")) }}
COSMOS_CURATOR_OTLP_METRICS_PUSH: {{ .Values.metrics.otlpPush.enabled | quote }}
{{- end }}
{{- if and .Values.metrics.otlpPush.enabled (hasKey .Values.metrics.otlpPush "interval") (not (hasKey $customEnv "COSMOS_CURATOR_OTLP_METRICS_PUSH_INTERVAL")) }}
COSMOS_CURATOR_OTLP_METRICS_PUSH_INTERVAL: {{ .Values.metrics.otlpPush.interval | quote }}
{{- end }}
{{- if and .Values.tracing.otlp.enabled (not (hasKey $customEnv "COSMOS_CURATOR_PROFILE_TRACING")) }}
COSMOS_CURATOR_PROFILE_TRACING: {{ .Values.tracing.otlp.enabled | quote }}
{{- end }}
{{- if and .Values.tracing.otlp.enabled (hasKey .Values.tracing.otlp "sampling") (not (hasKey $customEnv "COSMOS_CURATOR_PROFILE_TRACING_SAMPLING")) }}
COSMOS_CURATOR_PROFILE_TRACING_SAMPLING: {{ .Values.tracing.otlp.sampling | quote }}
{{- end }}
{{- if and $appOtlpEnabled (gt (len $otlpRunAttributes) 0) (not (hasKey $customEnv "COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES")) }}
COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES_VALUES: {{ include "curator-ray.otlp.runAttributes" . | quote }}
{{- end }}
{{- end }}

{{- define "curator-ray.validateOtlp" -}}
{{- $sharedOtlpEnabled := include "curator-ray.otlp.enabled" . | eq "true" -}}
{{- $otlpEndpoint := include "curator-ray.otlp.endpoint" . | trim -}}
{{- $metricsOtlpEndpoint := default $otlpEndpoint .Values.metrics.otlp.endpoint | trim -}}
{{- if and $sharedOtlpEnabled (not $otlpEndpoint) -}}
{{- fail "OTLP logs, traces, or in-process metrics push require otlp.endpoint or legacy metrics.otlp.endpoint" -}}
{{- end -}}
{{- if $sharedOtlpEnabled -}}
{{- range $signal := list "/v1/logs" "/v1/metrics" "/v1/traces" -}}
{{- if hasSuffix $signal (trimSuffix "/" $otlpEndpoint) -}}
{{- fail (printf "OTLP logs, traces, and in-process metrics push need a base endpoint without %s; signal exporters append that path themselves (got %q -- set otlp.endpoint explicitly when metrics.otlp.endpoint carries a full metrics URL)" $signal $otlpEndpoint) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if and $sharedOtlpEnabled (include "curator-ray.otlp.usesLegacyMetricsEndpoint" .) (or .Values.metrics.otlp.certPath .Values.metrics.otlp.keyPath) -}}
{{- fail "OTLP logs, traces, or in-process metrics push cannot use legacy metrics.otlp certPath/keyPath; set top-level otlp.endpoint and otlp.tls.* for pod-local mTLS" -}}
{{- end -}}
{{- if and .Values.metrics.otlp.enabled (not $metricsOtlpEndpoint) -}}
{{- fail "OTLP metrics export requires otlp.endpoint or metrics.otlp.endpoint" -}}
{{- end -}}
{{- if and (or .Values.metrics.otlpPush.enabled .Values.tracing.otlp.enabled) (contains "${env:" $otlpEndpoint) -}}
{{- fail "metrics.otlpPush.enabled or tracing.otlp.enabled cannot use collector-only ${env:...} substitution in otlp.endpoint" -}}
{{- end -}}
{{- $otlpCertDir := dir (include "curator-ray.otlp.certPath" .) -}}
{{- $otlpKeyDir := dir (include "curator-ray.otlp.keyPath" .) -}}
{{- $otlpCaPath := include "curator-ray.otlp.caPath" . -}}
{{- $otlpCertsMounted := and (include "curator-ray.otlp.enabled" . | eq "true") (or .Values.otlp.tls.secret.enabled .Values.otlp.extractNVCFSecrets) -}}
{{- if and $otlpCertsMounted (ne $otlpCertDir $otlpKeyDir) -}}
{{- fail "chart-managed OTLP TLS certPath and keyPath must share one directory" -}}
{{- end -}}
{{- if and $otlpCertsMounted $otlpCaPath (ne $otlpCertDir (dir $otlpCaPath)) -}}
{{- fail "chart-managed OTLP TLS caPath must share the certPath directory" -}}
{{- end -}}
{{- if and .Values.otlp.extractNVCFSecrets .Values.otlp.tls.caPath (not .Values.otlp.nvcfSecrets.caKey) -}}
{{- fail "otlp.tls.caPath requires otlp.nvcfSecrets.caKey when otlp.extractNVCFSecrets is enabled" -}}
{{- end -}}
{{- /* The sidecar can only receive files through the chart-managed cert volume:
       extraVolumes/extraVolumeMounts reach the curator container only. Rendering
       ca_file for a path nothing mounts would fail the collector at startup. */ -}}
{{- $chartRendersLogConfig := and .Values.logging.otlp.enabled (not .Values.logging.otlp.configMap.existingName) -}}
{{- if and $chartRendersLogConfig $otlpCaPath (not (or .Values.otlp.tls.secret.enabled .Values.otlp.extractNVCFSecrets)) -}}
{{- fail "otlp.tls.caPath needs a chart-managed source for the log sidecar; enable otlp.tls.secret.enabled or otlp.extractNVCFSecrets, or clear otlp.tls.caPath" -}}
{{- end -}}
{{- end }}
