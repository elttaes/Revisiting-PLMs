import argparse


def params_parser():
    parser = argparse.ArgumentParser(description="ESM-1b pretraining hyper-parameters")
    parser.add_argument(
        "--arch",
        default="MSA Transformer",
        type=str,
        help="model architecture",
    )
    parser.add_argument(
        "--final_bias",
        default=True,
        type=bool,
        help="final bias",
    )
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument(
        "--layers",
        default=12,
        type=int,
        metavar="N",
        help="number of layers"
    )
    parser.add_argument(
        "--embed_dim",
        default=768,
        type=int,
        metavar="N",
        help="embedding dimension"
    )
    parser.add_argument(
        "--logit_bias",
        action="store_true",
        help="whether to apply bias to logits"
    )
    parser.add_argument(
        "--ffn_embed_dim",
        default=3072,
        type=int,
        metavar="N",
        help="embedding dimension for FFN",
    )
    parser.add_argument(
        "--attention_heads",
        default=12,
        type=int,
        metavar="N",
        help="number of attention heads",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout to apply."
    )
    parser.add_argument(
        "--attention_dropout",
        default=0.1,
        type=float,
        help="Dropout to apply."
    )
    parser.add_argument(
        "--activation_dropout",
        default=0.1,
        type=float,
        help="Dropout to apply."
    )
    parser.add_argument(
        "--max_tokens",
        default=2 ** 14,
        type=int,
        help=(
            "Used during inference to batch attention computations in a single "
            "forward pass. This allows increased input sizes with less memory."
        ),
    )
    parser.add_argument(
        "--max_positions",
        default=1024,
        type=int,
        help="max positions",
    )
    return parser.parse_args()
