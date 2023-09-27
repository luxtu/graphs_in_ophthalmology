import os
import gzip
import shutil

def gzip_json(folder):
    """gzips all json files in a folder. The json files are replaced by json.gz files."""
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), 'rb') as f_in:
                with gzip.open(os.path.join(folder, filename + '.gz'), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(os.path.join(folder, filename))