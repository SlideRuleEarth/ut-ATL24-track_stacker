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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


def score_all(c, a, r, d, headers=False):

    if headers is True:
        print(f'Cls'
              f'\tName'
              f'\tAccuracy'
              f'\tWghtF1'
              f'\tMacroF1'
              f'\tMicroF1')

    # Get the scores
    p = d[a]
    acc = accuracy_score(r, p)
    weighted_f1 = f1_score(r, p, average="weighted")
    macro_f1 = f1_score(r, p, average="macro")
    micro_f1 = f1_score(r, p, average="micro")
    print(f'{c}'
          f'\t{a}'
          f'\t{acc:0.3f}'
          f'\t{weighted_f1:0.3f}'
          f'\t{macro_f1:0.3f}',
          f'\t{micro_f1:0.3f}')


def score_binary(c, a, ref, df, pos_label, headers=False):

    # Replace values
    r = ref.copy()
    d = df.copy()
    r[r != pos_label] = 0
    d[d != pos_label] = 0

    if headers is True:
        print(f'Cls'
              f'\tName'
              f'\tAccuracy'
              f'\tF1'
              f'\tBA'
              f'\tcalF1'
              f'\tMCC'
              f'\tavg4'
              )

    # Get the scores
    p = d[a]
    acc = accuracy_score(r, p)
    f1 = f1_score(r, p, pos_label=pos_label)
    ba = balanced_accuracy_score(r, p)
    mcc = matthews_corrcoef(r, p)
    cm = confusion_matrix(r, p)
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    r0 = 0.5
    cal_f1 = 2.0 * TPR / (TPR + (1.0 / r0) * FPR + 1)
    avg = (f1 + ba + cal_f1 + mcc) / 4.0
    print(f'{c}'
          f'\t{a}'
          f'\t{acc:0.3f}'
          f'\t{f1:0.3f}'
          f'\t{ba:0.3f}'
          f'\t{cal_f1:0.3f}'
          f'\t{mcc:0.3f}'
          f'\t{avg:0.3f}')


def main(args):

    # Get the filenames
    filenames = glob.glob(args.input_glob)

    if args.verbose:
        print(filenames, file=sys.stderr)
        print(f'{len(filenames)} total files', file=sys.stderr)

    df = pd.DataFrame()

    if args.ensemble_only:
        algorithms = [
            'ensemble',
            ]
    else:
        algorithms = [
            'qtrees',
            'cshelph',
            'medianfilter',
            'bathypathfinder',
            'openoceanspp',
            'coastnet',
            'pointnet',
            'ensemble',
            ]
        algorithms.sort()

    for n, fn in enumerate(filenames):

        if args.verbose:
            print(f'Reading {n + 1} of {len(filenames)}: {fn}',
                  file=sys.stderr)

        d = pd.read_csv(fn, engine='pyarrow')

        if args.verbose:
            print(f'Read {len(d.index)} rows', file=sys.stderr)

        d = d[['manual_label'] + algorithms]

        if args.verbose:
            print(d.columns, file=sys.stderr)

        df = pd.concat([df, d])

    if args.verbose:
        print(f'Final dataframe = {df.shape}', file=sys.stderr)

    ref = df['manual_label']

    if args.verbose:
        x = ref.unique()
        print(f'unique(ref): {x}', file=sys.stderr)

    # Replace NAN's with 0's
    df = df.fillna(0)
    # Replace 'unknown' with 'unclassified'
    df = df.replace(1.0, 0.0)
    # Replace 'water column' with 'unclassified'
    df = df.replace(45.0, 0.0)

    n_total = len(df.index)
    n_noise = len(df[df.manual_label == 0].index)
    n_bathy = len(df[df.manual_label == 40].index)
    n_surface = len(df[df.manual_label == 41].index)
    print(f'total photons {n_total}', file=sys.stderr)
    print(f'total noise photons {n_noise}', file=sys.stderr)
    print(f'total bathy photons {n_bathy}', file=sys.stderr)
    print(f'total surface photons {n_surface}', file=sys.stderr)

    for n, a in enumerate(algorithms):
        headers = True if n == 0 else False

        if args.verbose:
            print(f'Scoring {a}', file=sys.stderr)

        if args.all:
            score_all('all', a, ref, df, headers)
        else:
            # Remove photons labeled as surface
            df2 = df[df.manual_label != 41]
            ref2 = df2['manual_label']
            if args.verbose:
                print(f'Removed {n_total-len(df2.index)} surface photons',
                      file=sys.stderr)

            score_binary('surface', a, ref, df, 41, headers)
            score_binary('bathy', a, ref, df, 40)
            score_binary('nonsurface', a, ref2, df2, 40)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose',
                        action="store_true", default=False)
    parser.add_argument('-a', '--all', action='store_true',
                        help='Score all classes together')
    parser.add_argument('-e', '--ensemble_only',
                        action="store_true", default=False)
    parser.add_argument('input_glob',
                        type=str,
                        help='Input training filename glob')
    args = parser.parse_args()

    main(args)
