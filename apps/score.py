"""
Read a CSV containing columns of predictions and a reference, and
compute metrics.
"""

import argparse
import pandas as pd
import glob
import math
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def score_all(c, a, y, d, headers=False):

    if headers is True:
        print(f'Cls'
              f'\tName'
              f'\tAccuracy'
              f'\tWghtF1'
              f'\tMacroF1'
              f'\tMicroF1')

    # Get the scores
    p = d[a]
    acc = accuracy_score(y, p)
    weighted_f1 = f1_score(y, p, average="weighted")
    macro_f1 = f1_score(y, p, average="macro")
    micro_f1 = f1_score(y, p, average="micro")
    print(f'{c}'
          f'\t{a}'
          f'\t{acc:0.3f}'
          f'\t{weighted_f1:0.3f}'
          f'\t{macro_f1:0.3f}',
          f'\t{micro_f1:0.3f}')


def score_binary(c, a, y, df, pos_label, headers=False):

    # Replace values
    r = y.copy()
    d = df.copy()
    r[r != pos_label] = 0
    d[d != pos_label] = 0
    r[r != pos_label] = 1
    d[d != pos_label] = 1

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
    TN, FP, FN, TP = confusion_matrix(r, p).ravel()
    PP = TP + FP
    PN = TN + FN
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = 1.0 - FPR
    FNR = 1.0 - TPR
    PPV = TP / PP
    NPV = TN / PN
    FOR = FN / PN
    FDR = FP / PP
    acc = (TP + TN) / (TP + TN + FP + FN)
    f1 = (2*TP) / (2*TP + FP + FN)
    ba = (TPR + TNR) / 2.0
    mcc = math.sqrt(TPR * TNR * PPV * NPV) - math.sqrt(FNR * FPR * FOR * FDR)
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
            'bathypathfinder',
            'coastnet',
            'cshelph',
            'medianfilter',
            'openoceanspp',
            'qtrees',
            'ensemble',
            ]

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
        print(df.describe(), file=sys.stderr)

    y = df['manual_label'].copy()

    if args.verbose:
        print('Y=', file=sys.stderr)
        print(y.describe(), file=sys.stderr)
        print(f'unique(y): {y.unique()}', file=sys.stderr)

    # Score each algorithm
    for n, a in enumerate(algorithms):

        headers = True if n == 0 else False

        if args.verbose:
            print(f'Scoring {a}', file=sys.stderr)

        if args.all:
            score_all('all', a, y, df, headers)
        else:
            # Remove photons labeled as surface
            df2 = df[df.manual_label != 41].copy()
            y2 = df2['manual_label'].copy()
            if args.verbose:
                print(f'Removed {len(df)-len(df2.index)} surface photons',
                      file=sys.stderr)

            score_binary('surface', a, y, df, 41, headers)
            score_binary('bathy', a, y, df, 40)
            score_binary('nonsurface', a, y2, df2, 40)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose',
                        action="store_true", default=False)
    parser.add_argument('-a', '--all', action='store_true',
                        help='Score all classes together')
    parser.add_argument('-e', '--ensemble-only',
                        action="store_true", default=False)
    parser.add_argument('input_glob',
                        type=str,
                        help='Input training filename glob')
    args = parser.parse_args()

    main(args)
