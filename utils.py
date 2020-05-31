from collections import namedtuple
import yaml
import pathlib

CWD = pathlib.Path(__file__).parent.absolute()


def convert(dictionary: dict) -> namedtuple:
    """
    @param dictionary: dictionary containing hyper-parameters, can be nested
    @return: namedtuple of type HyperParameters
    """
    return namedtuple('HyperParameters', dictionary.keys())(**dictionary)


def load_params(path: str) -> namedtuple:
    """
    @param path: path of the yaml-file containing the hyper-parameters
    @return: namedtuple of type HyperParameters
    """
    with open(CWD / path) as file:
        param_dict = yaml.full_load(file)
    return convert(param_dict)


h_params = load_params("hyper_parameters.yml")
