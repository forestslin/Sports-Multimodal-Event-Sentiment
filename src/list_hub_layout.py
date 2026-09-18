"""Print public repository layouts without downloading feature arrays."""

from huggingface_hub import HfApi

api = HfApi()
for repo, revision in [
    ("SoccerNet/SN-echoes", "main"),
    ("SoccerNet/SN-Features", "resnet-tf2-pca512"),
]:
    print(f"\n--- {repo}@{revision} ---")
    for idx, item in enumerate(api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)):
        print(item.path)
        if idx == 19:
            break
