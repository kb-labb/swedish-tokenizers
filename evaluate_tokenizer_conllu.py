import sys
import os
import csv
import conllu
import numpy as np
from collections import Counter
from transformers import PreTrainedTokenizerFast

"""
- read conllu text and tokens without extra nodes x.xx
- also keep POS tag
- keep lemma
- count how many words are in vocab
- count how many lemmas are in vocab

- load tokenizer
- 
"""


def process_conllu(fn: str):
    lemmas = set()
    word2pos = {}
    texts = []
    with open(fn) as fh:
        for sentence in conllu.parse_incr(fh):
            text = sentence.metadata["text"]
            tok = [word["form"] for word in sentence if type(word["id"]) == int]
            texts.append((text, tok))
            for word in sentence:
                lemmas.add(word["lemma"])
                if word["form"] not in word2pos:
                    word2pos[word["form"]] = set()
                word2pos[word["form"]].add(word["upos"])
    return lemmas, word2pos, texts


def check_vocab(tokenizer, words, meta="▁"):
    vocab = set(tokenizer.get_vocab().keys())
    # print(len(vocab))
    prefixes = set()
    suffixes = set()
    for word in vocab:
        if word.startswith(meta):
            prefixes.add(word.lstrip(meta))
        else:
            suffixes.add(word)
    pcounter = 0
    scounter = 0
    bcounter = 0
    for word in words:
        if word in prefixes and word not in suffixes:
            pcounter += 1
        if word in suffixes and word not in prefixes:
            scounter += 1
        if word in prefixes and word in suffixes:
            bcounter += 1
    return {"voc_pref": len(prefixes),
            "voc_suff": len(suffixes),
            "w_in_p": pcounter,
            "w_in_s": scounter,
            "w_in_b": bcounter,
            }


def get_tok_text_lengths(tokenizer, orig_tok, pred_tok):
    real_lengths = [len(x) for x in orig_tok]
    tok_lengths = [len(x) for x in pred_tok]
    # tok_lengths = [len([y for y in x if y != "▁"]) for x in pred_tok]
    return real_lengths, tok_lengths


def evaluate_tokenizer(tokenizer_fn, data_fn):
    lemmas, word2pos, texts = process_conllu(data_fn)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_fn)
    results = {}
    tokenizer_fn = tokenizer_fn.replace(".very-swedish", "_very_swedish")
    train_data = tokenizer_fn.split("/")[-1].split(".")[0][4:]
    results["train_data"] = train_data.replace("-", "_")
    pretok = tokenizer_fn.split("/")[-1].split(".")[1][4:].split("-")[-1]
    results["pretok"] = pretok
    size = int(tokenizer_fn.split("/")[-1].split(".")[1][4:].split("-")[-2])
    results["size"] = size
    tok_type = "_".join(tokenizer_fn.split("/")[-1].split(".")[1][4:].split("-")[:-2])
    results["type"] = tok_type
    # print("lemmas")
    lemma_results = {"lemma_" + k: v for k, v in check_vocab(tokenizer, lemmas).items()}
    results.update(lemma_results)
    lemma_frtlt = sum(len(tokenizer.tokenize(x)) for x in lemmas) / len(lemmas)
    results["lemma_frtlt"] = lemma_frtlt
    # print(lemma_frtlt)
    # print("forms")
    forms = word2pos.keys()
    form_results = {"form_" + k: v for k, v in check_vocab(tokenizer, forms).items()}
    results.update(form_results)
    forms_frtlt = sum(len(tokenizer.tokenize(x)) for x in forms) / len(forms)
    results["form_frtlt"] = forms_frtlt

    # print(forms_frtlt)

    orig, otok = zip(*texts)
    ptok = [tokenizer.tokenize(doc) for doc in orig]
    cnt_otok = Counter(y for x in otok for y in x)
    cnt_ptok = Counter(y for x in ptok for y in x)
    results["otok_unique_tokens"] = len(cnt_otok)
    results["otok_total_tokens"] = sum(v for v in cnt_otok.values())
    results["otok_ttl/unq"] = sum(v for v in cnt_otok.values()) / len(cnt_otok)
    # print(len(cnt_otok),
    #       sum(v for v in cnt_otok.values()),
    #       sum(v for v in cnt_otok.values()) / len(cnt_otok),
    #       # cnt_otok.most_common(1),
    #       # np.mean(list(cnt_otok.values())),
    #       # np.median(list(cnt_otok.values())),
    #       # np.std(list(cnt_otok.values())),
    #       )
    results["ptok_unique_tokens"] = len(cnt_ptok)
    results["ptok_total_tokens"] = sum(v for v in cnt_ptok.values())
    results["ptok_ttl/unq"] = sum(v for v in cnt_ptok.values()) / len(cnt_ptok)
    # print(len(cnt_ptok),
    #       sum(v for v in cnt_ptok.values()),
    #       sum(v for v in cnt_ptok.values()) / len(cnt_ptok),
    #       # cnt_ptok.most_common(1),
    #       # np.mean(list(cnt_ptok.values())),
    #       # np.median(list(cnt_ptok.values())),
    #       # np.std(list(cnt_ptok.values())),
    #       )

    rl, tl = get_tok_text_lengths(tokenizer, otok, ptok)
    results["fertility"] = sum(tl) / sum(rl)
    # print(sum(tl) / sum(rl))
    return results


def main():
    tokenizers_dir = sys.argv[2]
    data_fn = sys.argv[1]
    with open(sys.argv[3], "w") as fout:
        first = True
        if os.path.isfile(tokenizers_dir):
            results = evaluate_tokenizer(tokenizers_dir, data_fn)
            csv.writer(fout).writerow(results.keys())
            csv.writer(fout).writerow(results.values())
            return

        for tok_fn in [x for x in os.listdir(tokenizers_dir) if x.endswith("json")]:
            results = evaluate_tokenizer(os.path.join(tokenizers_dir, tok_fn), data_fn)
            if first:
                csv.writer(fout).writerow(results.keys())
                first = False
            csv.writer(fout).writerow(results.values())




if __name__ == "__main__":
    # results = evaluate_tokenizer(sys.argv[2], sys.argv[1])
    main()

