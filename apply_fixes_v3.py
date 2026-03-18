import os

def apply_fixes():
    project_path = os.path.expanduser("~/projects/my_new_project/deep-live-cam-ui-fixed")

    # Fix ui.py import error
    ui_file_path = os.path.join(project_path, "modules", "ui.py")
    with open(ui_file_path, "r") as f:
        content = f.read()

    # Replace 'cluster_analysis' with 'find_cluster_centroids'
    content = content.replace("from modules.cluster_analysis import cluster_analysis", "from modules.cluster_analysis import find_cluster_centroids")

    with open(ui_file_path, "w") as f:
        f.write(content)
    print(f"Fixed import in {ui_file_path} successfully!")

if __name__ == "__main__":
    apply_fixes()
