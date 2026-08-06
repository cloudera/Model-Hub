#!/bin/bash
set -euo pipefail

# Regenerate all LoRA-affected 1.61.0 model catalogs
# Uses NGC_API_KEY from environment

cd "$(dirname "$0")/.."

SCRIPT="utils/generate-modelhub-catalog.py"
MANIFEST_BASE="manifests/1.19.0"
NGC_KEY="${NGC_API_KEY:-${NGC_CLI_API_KEY:-}}"
PUBLIC_GPU_WHITELIST="A10G L40S H100 H200 A100 B200 RTX6000_BLACKWELL_SV"

if [ -z "$NGC_KEY" ]; then
  echo "ERROR: NGC_API_KEY or NGC_CLI_API_KEY must be set"
  exit 1
fi

# Model definitions: output_file|platform|manifest1,manifest2,...
MODELS=(
  "models/private/1.61.0/gpt-oss.yaml|private|nim/openai/gpt-oss-20b.yaml,nim/openai/gpt-oss-120b.yaml"
  "models/private/1.61.0/llama-3.1-instruct.yaml|private|nim/meta/llama-3.1-8b-instruct-pb25h2.yaml,nim/meta/llama-3.1-70b-instruct-pb25h2.yaml"
  "models/private/1.61.0/llama-3.1-nemotron-nano-v1.yaml|private|nim/nvidia/llama3.1-nemotron-nano-4b-v1.1.yaml,nim/nvidia/llama-3.1-nemotron-nano-8b-v1.yaml"
  "models/private/1.61.0/llama-3.2-instruct.yaml|private|nim/meta/llama-3.2-1b-instruct.yaml,nim/meta/llama-3.2-3b-instruct.yaml"
  "models/private/1.61.0/llama-3.3-instruct.yaml|private|nim/meta/llama-3.3-70b-instruct.yaml"
  "models/private/1.61.0/llama-3.3-nemotron-super-49b.yaml|private|nim/nvidia/llama-3.3-nemotron-super-49b-v1.5.yaml"
  "models/private/1.61.0/mistral-instruct.yaml|private|nim/mistralai/mistral-7b-instruct-v0-3.yaml"
  "models/private/1.61.0/mixtral-instruct.yaml|private|nim/mistralai/mixtral-8x7b-instruct-v0-1.yaml"
  "models/private/1.61.0/nemotron-3-nano.yaml|private|nim/nvidia/nemotron-3-nano.yaml"
  "models/private/1.61.0/nemotron-3-super-120b-a12b.yaml|private|nim/nvidia/nemotron-3-super-120b-a12b.yaml"
  "models/public/1.61.0/gpt-oss.yaml|public|nim/openai/gpt-oss-20b.yaml,nim/openai/gpt-oss-120b.yaml"
  "models/public/1.61.0/llama-3.1-instruct.yaml|public|nim/meta/llama-3.1-8b-instruct-pb25h2.yaml,nim/meta/llama-3.1-70b-instruct-pb25h2.yaml"
  "models/public/1.61.0/llama-3.1-nemotron-nano-v1.yaml|public|nim/nvidia/llama3.1-nemotron-nano-4b-v1.1.yaml,nim/nvidia/llama-3.1-nemotron-nano-8b-v1.yaml"
  "models/public/1.61.0/llama-3.2-instruct.yaml|public|nim/meta/llama-3.2-1b-instruct.yaml,nim/meta/llama-3.2-3b-instruct.yaml"
  "models/public/1.61.0/llama-3.3-instruct.yaml|public|nim/meta/llama-3.3-70b-instruct.yaml"
  "models/public/1.61.0/llama-3.3-nemotron-super-49b.yaml|public|nim/nvidia/llama-3.3-nemotron-super-49b-v1.5.yaml"
  "models/public/1.61.0/mistral-instruct.yaml|public|nim/mistralai/mistral-7b-instruct-v0-3.yaml"
  "models/public/1.61.0/mixtral-instruct.yaml|public|nim/mistralai/mixtral-8x7b-instruct-v0-1.yaml"
  "models/public/1.61.0/nemotron-3-nano.yaml|public|nim/nvidia/nemotron-3-nano.yaml"
  "models/public/1.61.0/nemotron-3-super-120b-a12b.yaml|public|nim/nvidia/nemotron-3-super-120b-a12b.yaml"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r output platform manifests <<< "$entry"
  echo ""
  echo "=========================================="
  echo "Generating: $output (platform=$platform)"
  echo "=========================================="

  # Create temp base with optimizationProfiles: []
  tmp_base=$(mktemp /tmp/base_XXXXXX.yaml)
  python3 -c "
import sys
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 100000
with open('$output', 'r') as f:
    data = yaml.load(f)
for model in data.get('models', []):
    for variant in model.get('modelVariants', []):
        variant['optimizationProfiles'] = []
with open('$tmp_base', 'w') as f:
    yaml.dump(data, f)
"

  # Run generation for each manifest
  IFS=',' read -ra manifest_list <<< "$manifests"
  for manifest in "${manifest_list[@]}"; do
    manifest_path="$MANIFEST_BASE/$manifest"
    echo "  -> Processing manifest: $manifest_path"
    if [ "$platform" = "public" ]; then
      python3 "$SCRIPT" \
        --profiles-yaml "$manifest_path" \
        --base-model-file "$tmp_base" \
        --output-yaml "$tmp_base" \
        --ngc-api-key "$NGC_KEY" \
        --platform public \
        --whitelisted-gpus $PUBLIC_GPU_WHITELIST \
        --a100-max-count 1
    else
      python3 "$SCRIPT" \
        --profiles-yaml "$manifest_path" \
        --base-model-file "$tmp_base" \
        --output-yaml "$tmp_base" \
        --ngc-api-key "$NGC_KEY" \
        --platform private
    fi
  done

  # Copy result to final output
  cp "$tmp_base" "$output"
  rm -f "$tmp_base"
  echo "  Done: $output"
done

echo ""
echo "All models regenerated successfully."
