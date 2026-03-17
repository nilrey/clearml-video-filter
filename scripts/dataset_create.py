from clearml import StorageManager, Dataset
from pathlib import Path

dataset = Dataset.create(
    dataset_project="SPB Dataset", dataset_name="Base Dataset"
)

input_dir = "/home/sadmin/Documents/Mirror/data/input/"

for file_path in sorted(Path(input_dir).iterdir()):
    if file_path.is_file():
        dataset.add_files(path=str(file_path))

# Upload dataset to ClearML server (customizable)
dataset.upload()

# commit dataset changes
dataset.finalize()