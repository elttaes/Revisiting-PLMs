import argparse

def params_parser():
    parser = argparse.ArgumentParser(description="ESM-1b pretraining hyper-parameters")
    parser.add_argument(
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
        default=5120,
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
        default=1024,
        type=int,
        help="max positions",
    )
    parser.add_argument(
        "--msa_ids",
        default=15300,
        type=int,
        help="msa ids",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="local_rank",
    )
    # parser.add_argument(
    #     "--final_bias",
    #     default=True,
    #     type=bool,
    #     help="final bias",
    # )

    return parser.parse_args()

# def params_parser():
#     parser = argparse.ArgumentParser(description="ESM-1b pretraining hyper-parameters")
#     parser.add_argument(
#         "--layers", default=6, type=int, metavar="N", help="number of layers"
#     )
#     parser.add_argument(
#         "--embed_dim", default=768, type=int, metavar="N", help="embedding dimension"
#     )
#     parser.add_argument(
#         "--logit_bias", action="store_true", help="whether to apply bias to logits"
#     )
#     parser.add_argument(
#         "--ffn_embed_dim",
#         default=3072,
#         type=int,
#         metavar="N",
#         help="embedding dimension for FFN",
#     )
#     parser.add_argument(
#         "--dropout",
#         default=0.0,
#         type=float,
#         help="Dropout to apply."
#     )
#     parser.add_argument(
#         "--attention_heads",
#         default=12,
#         type=int,
#         metavar="N",
#         help="number of attention heads",
#     )
#     parser.add_argument(
#         "--arch",
#         default="roberta_large",
#         type=str,
#         help="model architecture",
#     )
#     parser.add_argument(
#         "--max_positions",
#         default=260,
#         type=int,
#         help="max positions",
#     )
#     parser.add_argument(
#         "--msa_ids",
#         default=15300,
#         type=int,
#         help="msa ids",
#     )
#     parser.add_argument(
#         "--local_rank",
#         default=0,
#         type=int,
#         help="local_rank",
#     )
#     # parser.add_argument(
#     #     "--final_bias",
#     #     default=True,
#     #     type=bool,
#     #     help="final bias",
#     # )

#     return parser.parse_args()