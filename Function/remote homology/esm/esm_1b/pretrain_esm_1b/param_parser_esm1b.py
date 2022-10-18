import argparse


def params_parser():
    parser = argparse.ArgumentParser(description="ESM-1b pretraining hyper-parameters")
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument(
        # 6
        "--layers", default=33, type=int, metavar="N", help="number of layers"
    )
    parser.add_argument(
        "--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension"
    )
    parser.add_argument(
        "--logit_bias", action="store_true", help="whether to apply bias to logits"
    )
    parser.add_argument(
        "--ffn_embed_dim",
        default=5120,  # 3072
        type=int,
        metavar="N",
        help="embedding dimension for FFN",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="Dropout to apply."
    )
    parser.add_argument(
        "--attention_heads",
        # 12
        default=20,
        type=int,
        metavar="N",
        help="number of attention heads",
    )
    parser.add_argument(
        "--arch",
        default="roberta_large",
        type=str,
        help="model architecture",
    )
    parser.add_argument(
        "--max_positions",
        default=260,
        type=int,
        help="max positions",
    )

    return parser.parse_args()
