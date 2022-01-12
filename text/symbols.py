""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
from itertools import chain

from utils import get_hparams

hps = get_hparams(init=False)

# Export all symbols:
symbols = list(
    chain.from_iterable(list(v) if k != "pad" else [v] for k, v in hps.symbols.items())
)

# Special symbol ids
SPACE_ID = symbols.index(" ")
