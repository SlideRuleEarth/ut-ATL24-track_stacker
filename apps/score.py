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


def score_all(algos, r, d):

    print(f'{"Algorithm":>20}'
          f'{"Accuracy":>10}'
          f'{"WghtF1":>10}'
          f'{"MacroF1":>10}')

    for a in algos:
        # Get the predictions for this algorithm
        p = d[a]
        # Get the scores
        acc = accuracy_score(r, p)
        weighted_f1 = f1_score(r, p, average="weighted")
        macro_f1 = f1_score(r, p, average="macro")
        print(f'{a:>20}'
              f'{acc:10.3f}'
              f'{weighted_f1:10.3f}'
              f'{macro_f1:10.3f}')


def score_surface(algos, r, d):

    # Replace 'bathy' with 'unclassified'
    r = r.replace(40.0, 0.0)
    d = d.replace(40.0, 0.0)

    print(f'{"Algorithm":>20}'
          f'{"Accuracy":>10}'
          f'{"F1":>10}'
          f'{"BA":>10}')

    for a in algos:
        # Get the predictions for this algorithm
        p = d[a]
        # Get the scores
        acc = accuracy_score(r, p)
        f1 = f1_score(r, p, pos_label=41.0)
        ba = balanced_accuracy_score(r, p)
        print(f'{a:>20}{acc:10.3f}{ba:10.3f}{f1:10.3f}')


def score_bathy(algos, r, d):

    # Replace 'surface' with 'unclassified'
    r = r.replace(41.0, 0.0)
    d = d.replace(41.0, 0.0)

    print(f'{"Algorithm":>20}'
          f'{"Accuracy":>10}'
          f'{"F1":>10}'
          f'{"BA":>10}')

    for a in algos:
        # Get the predictions for this algorithm
        p = d[a]
        # Get the scores
        acc = accuracy_score(r, p)
        f1 = f1_score(r, p, pos_label=40.0)
        ba = balanced_accuracy_score(r, p)
        print(f'{a:>20}{acc:10.3f}{ba:10.3f}{f1:10.3f}')


def main(args):

    # Show args
    if args.verbose:
        print(args, file=sys.stderr)

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

        if 'qtrees' in d.columns:
            QTREES_LABEL = 'qtrees'
        else:
            QTREES_LABEL = 'prediction'
        d = d[['index_ph',
               QTREES_LABEL,
               'cshelph',
               'medianfilter',
               'bathypathfinder',
               'openoceans',
               'openoceanspp',
               'coastnet',
               'pointnet',
               'manual_label']]
        d.rename(columns={"prediction": "qtrees"})

        if args.verbose:
            print(d.columns)

        df = pd.concat([df, d])

    if args.verbose:
        print(f'Final dataframe = {df.shape}')

    algorithms = [
        'qtrees',
        'cshelph',
        'medianfilter',
        'bathypathfinder',
        'openoceans',
        'openoceanspp',
        'coastnet',
        'pointnet',
        ]
    algorithms.sort()

    ref = df['manual_label']

    if args.verbose:
        x = ref.unique()
        print(f'unique(ref): {x}')
        for col in algorithms:
            x = df[col].unique()
            print(f'unique({col}): {x}')

    # Replace NAN's with 0's
    df = df.fillna(0)
    # Replace 'unknown' with 'unclassified'
    df = df.replace(1.0, 0.0)
    # Replace 'water column' with 'unclassified'
    df = df.replace(45.0, 0.0)

    if args.verbose:
        print('All labels')

    score_all(algorithms, ref, df)

    if args.verbose:
        print('Surface labels')

    score_surface(algorithms, ref, df)

    if args.verbose:
        print('Bathy labels')

    score_bathy(algorithms, ref, df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", default=False)
    parser.add_argument('input_glob',
                        type=str,
                        help='Input training filename glob')
    args = parser.parse_args()

    main(args)
