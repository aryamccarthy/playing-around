"""
Make dummy morphological sentences to improve the char embedding in taggers.
"""

import argparse
from argparse import Namespace

from collections import namedtuple

from pathlib import Path

UniMorphRow = namedtuple('UniMorphRow', 'lemma form bundle')

UNIMORPH_FORMAT_STR = "1\t{form}\t{lemma}\t{pos}\t-\t{bundle}\t-\t-\t-\t-"


def stream_as_tuples(file: Path):
    with open(file) as f:
        for line in f:
            # Ignore multi-word, since we can't tag it.
            try:
                assert ' ' not in line
                yield UniMorphRow._make(line.strip().split("\t"))
            except (AssertionError, TypeError):  # Expected 3 arguments, got 1
                continue


def process_file(file: Path) -> None:
    with open(file) as f:
        for i, unimorph in enumerate(stream_as_tuples(file)):
            print(f"# sent_id = {str(file)}-{i}")
            print(f"# text = {unimorph.form}")
            pos_tag = unimorph.bundle.split(";")[0]
            if pos_tag == "N":
                pos_tag = "NOUN"
            elif pos_tag in {"V", "V.PTCP", "V.CVB", "V.MSDR"}:
                pos_tag = "VERB"
            else:
                assert pos_tag == "ADJ", pos_tag
            print(UNIMORPH_FORMAT_STR.format(
                    form=unimorph.form,
                    pos=pos_tag,
                    lemma=unimorph.lemma,
                    bundle=unimorph.bundle,
                ))
            print()


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=Path)
    return parser.parse_args()

def main():
    args = parse_args()
    process_file(args.infile)


if __name__ == '__main__':
    main()