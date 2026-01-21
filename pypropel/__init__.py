import sys

from . import (
    dist,
    dataset,
    msa,
    eval,
    # fcsdf,
    fpmsa,
    fpseq,
    fpstr,
    fpsite,
    external,
    convert,
    io,
    plot,
    seq,
    str,
    uniprot,
    mol,
    gvp,
    graph,
)

# ESM is optional (requires fair-esm)
try:
    from . import esm
except ImportError:
    pass