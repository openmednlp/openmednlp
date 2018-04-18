import configparser
from collections import namedtuple


def config_to_namedtuple(config_path='config.ini'):
    _config = configparser.ConfigParser()
    _config.read(config_path)
    groups_dict = dict()
    for group_name in _config.keys():
        groups_dict[group_name] = namedtuple(group_name, _config[group_name].keys())(**_config[group_name])
    return namedtuple('Config', _config.keys())(**groups_dict)


constants = config_to_namedtuple()
