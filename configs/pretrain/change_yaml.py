import yaml
import os
import glob
from collections import OrderedDict

class OrderedLoader(yaml.SafeLoader):
    pass

def construct_ordered_mapping(loader, node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))

OrderedLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    construct_ordered_mapping
)

class OrderedDumper(yaml.SafeDumper):
    pass

def represent_ordered_mapping(dumper, data):
    return dumper.represent_dict(data.items())

OrderedDumper.add_representer(OrderedDict, represent_ordered_mapping)

pretrain_yaml_folder = "./"
file_extension = "experiment.yaml"

yaml_files = glob.glob(f"{pretrain_yaml_folder.rstrip("/")}/**/*{file_extension}", recursive=True)

for file in yaml_files:
    params = None
    print(file)
    with open(file, 'r') as y_file:
        params = yaml.load(y_file, Loader=OrderedLoader)
        
    # params["logging"]["folder"] = "logs/"
    params["meta"]["ckpt_folder"] = "src/models/pretrained_weights/"
        
    with open(file, 'w') as file:
        yaml.dump(params, file, Dumper=OrderedDumper, default_flow_style=False)  # Set default_flow_style to False for human-readable format