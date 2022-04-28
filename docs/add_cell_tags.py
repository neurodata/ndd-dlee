import nbformat as nbf
from glob import glob
import numpy as np

# Collect a list of all notebooks in the content folder
notebooks = glob("./*.ipynb", recursive=True)

# Search through each notebook and add hide-input tag
for ipath in notebooks:
    ntbk = nbf.read(ipath, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        if cell.get("cell_type") == "code":
            cell_tags = cell.get("metadata", {}).get("tags", [])
            cell_tags = list(np.unique(cell_tags))
            if "hide-input" not in cell_tags:
                cell_tags.append("hide-input")

    nbf.write(ntbk, ipath)

print('Add cell tag')