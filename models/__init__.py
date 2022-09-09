from importlib import import_module


def make_model(args):
    module = import_module("models." + args.model.lower())
    return module.make_model(args)
