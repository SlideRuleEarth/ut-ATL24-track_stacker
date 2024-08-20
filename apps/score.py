"""
Read a CSV containing columns of predictions and a reference, and
compute metrics.
"""

import argparse
import pandas as pd
import glob
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score


def score_all(r, d):

    print(f'{"Accuracy":>10}'
          f'{"WghtF1":>10}'
          f'{"MacroF1":>10}')

    # Get the scores
    p = d['prediction']
    acc = accuracy_score(r, p)
    weighted_f1 = f1_score(r, p, average="weighted")
    macro_f1 = f1_score(r, p, average="macro")
    print(f'{acc:10.3f}'
          f'{weighted_f1:10.3f}'
          f'{macro_f1:10.3f}')


def score_surface(r, d):

    # Replace 'bathy' with 'unclassified'
    r = r.replace(40.0, 0.0)
    d = d.replace(40.0, 0.0)

    print(f'{"Accuracy":>10}'
          f'{"F1":>10}'
          f'{"BA":>10}')

    # Get the scores
    p = d['prediction']
    acc = accuracy_score(r, p)
    f1 = f1_score(r, p, pos_label=41.0)
    ba = balanced_accuracy_score(r, p)
    print(f'{acc:10.3f}{ba:10.3f}{f1:10.3f}')


def score_bathy(r, d):

    # Replace 'surface' with 'unclassified'
    r = r.replace(41.0, 0.0)
    d = d.replace(41.0, 0.0)

    print(f'{"Accuracy":>10}'
          f'{"F1":>10}'
          f'{"BA":>10}')

    # Get the scores
    p = d['prediction']
    acc = accuracy_score(r, p)
    f1 = f1_score(r, p, pos_label=40.0)
    ba = balanced_accuracy_score(r, p)
    print(f'{acc:10.3f}{ba:10.3f}{f1:10.3f}')


def main(args):

    # Get the filenames
    filenames = glob.glob(args.input_glob)

    if args.verbose:
        print(filenames, file=sys.stderr)
        print(f'{len(filenames)} total files', file=sys.stderr)

    df = pd.DataFrame()

    for n, fn in enumerate(filenames):

        if args.verbose:
            print(f'Reading {n + 1} of {len(filenames)}: {fn}')

        d = pd.read_csv(fn, engine='pyarrow')

        if args.verbose:
            print(f'Read {len(d.index)} rows')

        d = d[['manual_label', 'prediction']]

        if args.verbose:
            print(d.columns)

        df = pd.concat([df, d])

    if args.verbose:
        print(f'Final dataframe = {df.shape}')

    ref = df['manual_label']

    if args.verbose:
        x = ref.unique()
        print(f'unique(ref): {x}')

    # Replace NAN's with 0's
    df = df.fillna(0)
    # Replace 'unknown' with 'unclassified'
    df = df.replace(1.0, 0.0)
    # Replace 'water column' with 'unclassified'
    df = df.replace(45.0, 0.0)

    if args.verbose:
        print('All labels')

    score_all(ref, df)

    if args.verbose:
        print('Surface labels')

    score_surface(ref, df)

    if args.verbose:
        print('Bathy labels')

    score_bathy(ref, df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", default=False)
    parser.add_argument('input_glob',
                        type=str,
                        help='Input training filename glob')
    args = parser.parse_args()

    main(args)
