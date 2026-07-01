from importlib import import_module

__all__ = [
    'dist',
    'dataset',
    'msa',
    'eval',
    'fpmsa',
    'fpseq',
    'fpstr',
    'fpsite',
    'external',
    'convert',
    'io',
    'plot',
    'seq',
    'str',
    'uniprot',
    'qc',
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
