import argparse
import json
import math
import os
import sys
from itertools import chain

import numpy as np
import torch
from g2p import make_g2p
from scipy.io.wavfile import write
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import (TextAudioCollate, TextAudioLoader,
                        TextAudioSpeakerCollate, TextAudioSpeakerLoader)
from features import get_features
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_feats(text, hps, lang):
    G2P = make_g2p(lang, f'{lang}-ipa')
    # multilingual case
    if lang in hps.symbols:
        lang_symbols = hps.symbols[lang]
    else:
        lang_symbols = hps.symbols
    symbols = list(chain.from_iterable(list(v) if k != "pad" else [v] for k, v in lang_symbols.items())
    )
    TOKENIZER = utils.create_tokenizer(symbols)
    text_norm = G2P(text).output_string
    tokenized_text = TOKENIZER.tokenize(text_norm)
    feats = np.array(get_features(tokenized_text))
    if hps.data.add_blank:
        zeros = np.zeros_like(feats)
        feats = np.hstack([zeros, feats]).reshape(feats.shape[0]*2, hps.data.n_feats)
        feats = np.pad(feats, [(0,1),(0,0)]) # add zeros at end of t
    feats = torch.from_numpy(feats).float()
    return feats
   
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
parser.add_argument('-m', '--model', type=str, required=False,
                      help='Model name')
parser.add_argument('-t', '--text', type=str, required=True, help='Text to synthesize')
parser.add_argument('-l', '--lang', type=str, required=False, help='Lang to synthesize')

args = parser.parse_args()

hps_config = args.config
ckpt = args.model

hps = utils.get_hparams_from_file(hps_config)

if hps.data.n_feats > 0:
    vocab = hps.data.n_feats
else:
    vocab = len(symbols)

net_g = SynthesizerTrn(
    vocab,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint(ckpt, net_g, None)
if hps.model.use_pfs:
    stn_tst = get_feats(args.text, hps, args.lang)
else:
    stn_tst = get_text(args.text, hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
write('test.wav', hps.data.sampling_rate, audio)
