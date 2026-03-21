from huggingface_hub import HfApi, login

api = HfApi()

# I ran this from inside my 'saved' folder. you can 'cd' into your location and then run this or adjust the 'folder_path'
api.upload_folder(
    folder_path="models/tsfm/finetuned",
    repo_id="umass-lass/fmtk-decoder-zoo",
    commit_message="tsfm decoders"
)