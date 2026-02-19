import nbformat
import sys
from pathlib import Path

def fix_notebook(input_path, output_path=None):
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path  # overwrite safely

    # Read notebook
    nb = nbformat.read(str(input_path), as_version=4)

    # Remove broken widgets metadata ONLY
    if "widgets" in nb.metadata:
        print("Removing corrupted metadata.widgets...")
        del nb.metadata["widgets"]
    else:
        print("No metadata.widgets found. Nothing to fix.")

    # Write notebook back (preserves outputs, images, everything else)
    nbformat.write(nb, str(output_path))
    print(f"Notebook successfully fixed: {output_path}")

if __name__ == "__main__":
    fix_notebook("Detection & Localization Notebook.ipynb")