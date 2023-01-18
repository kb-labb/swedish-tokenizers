from tabulate import tabulate
# tokenizers/tok-sample.txt-wordpiece-32000-pretok.json sample.txt
# Total tokens: 261970
# Unique tokens: 18216
# Average token usage: 14.381313131313131
# Median token usage: 4.0
# Standard deviation token usage: 139.98688442372165
# Average document length whitespace: 184.97230769230768
# Average document length tokenized: 268.6871794871795
# Average fertility: 1.4986815805779305
# tokenizers/tok-sample.txt-wordpiece-32000-pretok.json sample-10.txt

keymap = {
          "Total tokens": "ttl tok",
          "Unique tokens": "unq tok",
          "Average token usage": "avg tok usg",
          "Median token usage": "med tok usg",
          "Standard deviation token usage": "std tok usg",
          "Average document length whitespace": "avg doc len ws",
          "Average document length tokenized": "avg doc len tok",
          "Average fertility": "avg frtlty",
          }


def read_entry(line, fin):
    entry = {}
    name, data = line.split()
    name = name.split("/")[1]
    print(name)
    name = name.replace("-10", "_10")
    name = name.replace("-again", "_again")
    name = name.replace("bpe-straight", "bpe_straight")
    name = name.replace("spe-bpe", "spe_bpe")
    name = name.split("-")
    train_data = name[1]
    tok_type = name[2]
    size = int(name[3])
    pretok = name[4].split(".")[0]
    entry["type"] = tok_type
    entry["train data"] = train_data
    entry["size"] = size
    entry["pretok"] = pretok
    entry["eval data"] = data
    line = next(fin)
    while ":" in line:
        key, value = line.strip().split(":")
        key = keymap[key]
        if "." not in value:
            entry[key] = int(value)
        else:
            entry[key] = float(value)
        line = next(fin)
    return entry, line


def parse_tatt(fn):
    entries = []
    with open(fn) as fin:
        line = next(fin)
        while line:
            while line.startswith("tokenizers/"):
                new_entry, line = read_entry(line, fin)
                entries.append(new_entry)

            try:
                line = next(fin)
            except StopIteration:
                return entries


def main(fn):
    entries = parse_tatt(fn)
    print(tabulate(entries, headers="keys", tablefmt="pipe"))
    with open(".".join(fn.split(".")[:-1]) + ".tsv", "w") as fout:
        print(tabulate(entries, headers="keys", tablefmt="tsv"), file=fout)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
