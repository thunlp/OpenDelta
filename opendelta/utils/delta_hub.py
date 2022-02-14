

def create_hub_repo_name(root = "DeltaHub",
                         dataset = None,
                         delta_type = None,
                         model_name_or_path = None,
                         ):
    r"""Currently, it's only a simple concatenation of the arguments. 
    """
    repo_name = []

    repo_name.append(f"{delta_type}")
    model_name_or_path = model_name_or_path.split("/")[-1]
    repo_name.append(f"{model_name_or_path}")
    repo_name.append(f"{dataset}")

    repo_name = "_".join(repo_name)

    repo_name = root+"/"+repo_name
    return repo_name




