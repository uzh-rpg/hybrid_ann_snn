import glob
import os
import re


def get_file_paths(path, extensions):
    extension_regex = "|".join(extensions)

    files = []

    for dir_name, _, _ in os.walk(path):

        files.extend(
            [
                os.path.join(dir_name, f)
                for f in os.listdir(dir_name)
                if re.search(r'({})$'.format(extension_regex), f)
            ]
        )

    return sorted(files)
