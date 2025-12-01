from huggingface_hub import snapshot_download

local_dir = snapshot_download(
  repo_id="amcl6120/open-voice-detect-fast",
  local_dir="."
)
