import json
import logging

import torch


class DotDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __contains__(self, item):
        return item in self.keys()


class ConfigBase:
    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
        for key, value in params.items():
            setattr(self, key, self._to_dotdict(value))


class Params(ConfigBase):
    def __init__(self, json_path=None):
        if json_path:
            self.load(json_path)

    def _to_dotdict(self, obj):
        if isinstance(obj, dict):
            return DotDict({k: self._to_dotdict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [self._to_dotdict(item) for item in obj]
        else:
            return obj

    def load(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
        self.__dict__ = self._to_dotdict(params)

    def add(self, key, value):
        if "ADDED" not in self.__dict__:
            self.ADDED = DotDict()
        self.ADDED[key] = value


class RunningAverageDict:
    def __init__(self, keys):
        self.total_dict = {}
        for key in keys:
            self.total_dict[key] = 0
        self.steps = 0

    def update(self, val_dict):
        for key in self.total_dict:
            if key.endswith(":c"):
                self.total_dict[key] = val_dict[key]
            else:
                self.total_dict[key] += val_dict[key]
        self.steps += 1

    def serialize(self):
        keys = list(self.total_dict.keys())
        values = torch.tensor([list(self.total_dict.values())], dtype=torch.float32)
        steps = torch.tensor([self.steps], dtype=torch.float32)

        return keys, values, steps

    def reset(self):
        for key in self.total_dict:
            self.total_dict[key] = 0
        self.steps = 0

    def __call__(self):
        return {
            key.split(":")[0] if key.endswith(":c") else key: (
                value if key.endswith(":c") else value / float(self.steps)
            )
            for key, value in self.total_dict.items()
        }


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


def clear_handlers(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def save_dict_to_json(d, json_path):
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
