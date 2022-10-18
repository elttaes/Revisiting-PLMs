import argparse

import torch
from esm.data import Alphabet
from argparse import ArgumentParser
from esm.model import ProteinBertModel, MSATransformer


def TinyEsm1b_config():
    alphabet = Alphabet.from_architecture('ESM-1b')
    parser = ArgumentParser()
    parser.add_argument(
        "--arch", default="roberta_large"
    )
    parser.add_argument(
        "--layers", default=12, type=int
    )
    parser.add_argument(
        "--max_positions", default=1024, type=int
    )
    parser.add_argument(
        "--embed_dim", default=640, type=int, metavar="N", help="embedding dimension"
    )
    parser.add_argument(
        "--emb_layer_norm_before", default=True, type=bool
    )
    parser.add_argument(
        "--logit_bias", action="store_true", help="whether to apply bias to logits"
    )
    parser.add_argument(
        "--ffn_embed_dim",
        default=2560,
        type=int,
        metavar="N",
        help="embedding dimension for FFN",
    )
    parser.add_argument(
        "--attention_heads",
        default=20,
        type=int,
        metavar="N",
        help="number of attention heads",
    )

    return parser.parse_args(), alphabet


def ESM1b_config():
    alphabet = Alphabet.from_architecture('ESM-1b')
    params = {'arch': 'roberta_large',
              'layers': 33,
              'max_positions': 1024,
              'embed_dim': 1280,
              'emb_layer_norm_before': True,
              'ffn_embed_dim': 5120,
              'attention_heads': 20}

    return argparse.Namespace(**params), alphabet


def MSA_config():
    alphabet = Alphabet.from_architecture('MSA Transformer')
    params = {'layers': 12,
              'embed_dim': 768,
              'ffn_embed_dim': 3072,
              'attention_heads': 12,
              'dropout': 0.1,
              'attention_dropout': 0.1,
              'activation_dropout': 0.1,
              'max_tokens': 2 ** 14,
              'max_positions': 1024,
              'embed_positions_msa': True}

    return argparse.Namespace(**params), alphabet


def load_model_architecture(name):
    if name == 'esm-1b':
        args, alphabet = ESM1b_config()
        model = ProteinBertModel(args, alphabet)
        return model

    elif name == 'msa':
        args, alphabet = MSA_config()
        model = MSATransformer(args, alphabet)
        return model

    elif name == 'esm-1b-tiny':
        args, alphabet = TinyEsm1b_config()
        model = ProteinBertModel(args, alphabet)
        return model