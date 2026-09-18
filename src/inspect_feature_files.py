"""Inspect one public feature game directory before selecting a cohort."""

from huggingface_hub import HfApi

game = "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
api = HfApi()
for item in api.list_repo_tree(
    "SoccerNet/SN-Features",
    repo_type="dataset",
    revision="resnet-tf2-pca512",
    path_in_repo=game,
    recursive=True,
):
    print(item.path, getattr(item, "size", None))
