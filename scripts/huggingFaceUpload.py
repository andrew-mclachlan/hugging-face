from huggingface_hub import upload_folder

upload_folder(
    folder_path="./",
    repo_id="adi/open-voice-detect",
    repo_type="model",
    commit_message="Initial upload",
    revision="1"
)
