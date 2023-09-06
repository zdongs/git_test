import pickle
import tarfile
from word_1_Corpus_conversion import line_w


def main(input_file="data/people-2014.tar.gz", out_file="data/corpus-2014.data"):
    tar = tarfile.open((input_file), "r:gz", encoding="utf-8")
    corpus = []

    for tarinfo in tar:
        # 只读取文件内容
        if tarinfo.isreg():
            f = tar.extractfile(tarinfo)
            ctx = f.read()
            ctx = ctx.decode("utf-8").strip().split()
            ctx = [c.split(sep="/")[0].replace("[", "").replace("]", "") for c in ctx]
            if ctx:
                t1, t2 = line_w(ctx)
                corpus.append((t1, t2))

    with open(out_file, "wb") as f:
        pickle.dump(corpus, f)
    return corpus


if __name__ == "__main__":
    corpus = main()
