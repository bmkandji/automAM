import os


def set_project_root():
    """
    Sets the current working directory to the root of the 'automAM' project,
    assuming this script is in a subdirectory off the root.
    """
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)

    # Calculate the project root,
    # and the project structure looks like:
    # automAM/
    #   configs/
    #     root_config.py
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    # Change the current working directory to the project root
    os.chdir(project_root)
    print(f"Working directory set to the project root: {project_root}")
