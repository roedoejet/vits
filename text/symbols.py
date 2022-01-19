""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
from itertools import chain

from utils import get_hparams

hps = get_hparams()

# Export all symbols:
if hps.data.n_langs > 1:
    symbols = {
        lang_k: list(
            chain.from_iterable(list(v) if k != "pad" else [v] for k, v in lang_v.items())
        ) for lang_k, lang_v in hps.symbols.items()
        }
else:
    symbols = list(
        chain.from_iterable(list(v) if k != "pad" else [v] for k, v in hps.symbols.items())
    )
