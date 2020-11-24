import json
from toolbox.file_system_tools import FSL
from pathlib import Path

# load up
INFO = json.loads(FSL.read("../settings/info.json"))

def process_paths_mapping(source, dict_mapping):
    resulting_paths = {}
    # make paths absolute if they're relative
    for each_key, each_value in dict_mapping.items():
        # only considered if name is a string
        if type(each_key) is str:
            # if its a path then convert it
            if type(each_value) is str:
                *folders, name, ext = FSL.path_pieces(each_value)
                
                # if there are no folders then it must be a relative path
                # (otherwise it would start with the root "/" folder)
                if len(folders) == 0:
                    folders.append(".")
                
                # if not absolute, then make it absolute
                if folders[0] != "/":
                    if folders[0] == '.' or folders[0] == './':
                        _, *folders = folders
                    resulting_paths[each_key] = Path(FSL.join(source, each_value))
                else:
                    resulting_paths[each_key] = Path(each_value)
    
    return resulting_paths

PATHS = process_paths_mapping(FSL.pwd(), INFO["paths"])
PARAMETERS = INFO["parameters"]