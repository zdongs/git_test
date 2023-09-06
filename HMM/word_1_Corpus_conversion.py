import pickle

# tokens, tags = [], []

# for line in open("data\pku_training.utf8", "r", encoding="utf-8"):
#     words = line.strip().split()
#     t1, t2 = [], []
#     for w in words:
#         t1.extend(list(w))

#         if len(w) == 1:
#             t2.append("S")
#         else:
#             t2.append("B")
#             t2.extend(["M"] * (len(w) - 2))
#             t2.append("E")
#     tags.append(t2)
#     tokens.append(t1)

# with open("data\corpus.data", "wb") as f:
#     pickle.dump((tokens, tags), f)


def line_w(line):
    t1, t2 = [], []
    for w in line:
        t1.extend(list(w))
        if len(w) == 1:
            t2.append("S")
        else:
            t2.append("B")
            t2.extend(["M"] * (len(w) - 2))
            t2.append("E")
    return t1, t2


def main(input_file="data/pku_training.utf8", output_file="data/corpus.data"):
    corpus = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            if line:
                t1, t2 = line_w(line)
                corpus.append((t1, t2))

    with open(output_file, "wb") as f:
        pickle.dump(corpus, f)


if __name__ == "__main__":
    main()
