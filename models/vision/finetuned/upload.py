from huggingface_hub import HfApi, login

# 1. Log in with your Write Token
# (It will ask for the token when you run it, or you can paste it here)


api = HfApi()

# 2. Upload the 'saved' folder
# This will keep /model-x/decoder.pth exactly as it is locally
api.upload_folder(
    folder_path="./",
    repo_id="umass-lass/fmtk-decoder-zoo",
    commit_message="dino nyu depth estimation"
)

print("Upload Complete! Check your repo 'Files' tab.")