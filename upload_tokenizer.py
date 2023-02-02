from transformers import PreTrainedTokenizerFast
import argparse


def load_tokenizer(fn, bert_style):
    if bert_style:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(fn,
                                                            model_max_length=2048,
                                                            unk_token="[UNK]",
                                                            pad_token="[PAD]",
                                                            bos_token="[CLS]",
                                                            eos_token="[SEP]",
                                                            cls_token="[CLS]",
                                                            sep_token="[SEP]",
                                                            mask_token="[MASK]",
                                                            )
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(fn,
                                                            model_max_length=2048,
                                                            unk_token="<unk>",
                                                            pad_token="<pad>",
                                                            bos_token="<bos>",
                                                            eos_token="<eos>",
                                                            cls_token="<bos>",
                                                            sep_token="<eos>",
                                                            mask_token="<mask>",
                                                            )
    return tokenizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--style", choices=["bert", "gpt"])

    return parser.parse_args()


def main():

    args = get_args()

    tokenizer = load_tokenizer(args.source, args.style)

    tokenizer.push_to_hub(args.target)

    return


if __name__ == "__main__":
    main()
