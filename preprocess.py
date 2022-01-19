
import argparse
import os
from itertools import chain

import numpy as np
from g2p import make_g2p
from tqdm import tqdm

import text
from features import get_features
from utils import create_tokenizer, get_hparams, load_filepaths_and_text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--spkr_index", default=None, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["basic_cleaners"])
  parser.add_argument("--use_pfs", action='store_true')
  parser.add_argument("--pf_lang", default=None, type=str)
  parser.add_argument("--lang_index", default=None, type=int)
  parser.add_argument("--is_ipa", action='store_true')

  args, unknown = parser.parse_known_args()

  G2P_DICT = {}
  
  if args.use_pfs:
    hps = get_hparams()
    if not os.path.exists('feats'):
      os.mkdir('feats')
    elif args.lang_index and hps.model.n_langs > 1:
      symbols = { lang_k: list(
        chain.from_iterable(list(v) if k != "pad" else [v] for k, v in lang_v.items())
      ) for lang_k, lang_v in hps.symbols.items()}
      TOKENIZERS = {k: create_tokenizer(v) for k,v in symbols.items()}
    elif args.pf_lang:
      symbols = list(
        chain.from_iterable(list(v) if k != "pad" else [v] for k, v in hps.symbols.items())
      )
      TOKENIZERS = {args.pf_lang: create_tokenizer(symbols)}
    else:
      print("If using pf inputs, must provide index for multilingual model or pf_lang")
      exit()


  for filelist in tqdm(args.filelists):
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in tqdm(range(len(filepaths_and_text))):
      original_text = filepaths_and_text[i][args.text_index]
      cleaned_text = text._clean_text(original_text, args.text_cleaners)
      filepaths_and_text[i][args.text_index] = cleaned_text
      if args.use_pfs:
        if args.pf_lang:
          input_lang = args.pf_lang
        elif args.lang_index:
          input_lang = filepaths_and_text[i][args.lang_index]
        else:
          print("If using phonological features, you must define the input language that must be compatible with g2p")
          exit()
        
        if args.is_ipa:
          ipa_text = cleaned_text
        else:
          if input_lang not in G2P_DICT:
            G2P_DICT[input_lang] = make_g2p(input_lang, f'{input_lang}-ipa')
          ipa_text = G2P_DICT[input_lang](cleaned_text).output_string
        
        if args.spkr_index:
          speaker = filepaths_and_text[i][args.spkr_index]
        else:
          speaker = 0

        if args.lang_index:
          lang = filepaths_and_text[i][args.lang_index]
        else:
          lang = 0
        
        basename, _ = os.path.splitext(os.path.basename(filepaths_and_text[i][0]))

        feat_filename = "{}-{}-feat-{}.npy".format(lang, speaker, basename)

        feats = get_features(TOKENIZERS[input_lang].tokenize(ipa_text))

        np.save(
            os.path.join("feats", feat_filename),
            feats,
        )
        

    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
