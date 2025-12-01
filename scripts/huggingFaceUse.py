from huggingface_hub import HfApi

HF_TOKEN="csa_570dfb9aec60960846fd756bae039ejlrmWw"
HF_ENDPOINT="https://huggingface.cloudsmith.io/adi/ml-huggingface-test"

api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)

local_dir = api.snapshot_download(
    repo_id="adi/open-voice-detect",
    local_dir="./my-local-model-folder"
)
