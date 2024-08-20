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


def score_binary(r, d, pos_label):

    # Replace values
    r[r != pos_label] = 0
    d[d != pos_label] = 0

    print(f'{"Accuracy":>10}'
          f'{"F1":>10}'
          f'{"BA":>10}'
          f'{"calF1":>10}'
          f'{"MCC":>10}'
          f'{"avg4":>10}'
          )

    # Get the scores
    p = d['prediction']
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
    print(f'{acc:10.3f}{f1:10.3f}{ba:10.3f}{cal_f1:10.3f}{mcc:10.3f}{avg:10.3f}')


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

    print('All labels')
    score_all(ref, df)

    print('Surface labels')
    score_binary(ref.copy(), df.copy(), 41)

    print('Bathy labels')
    score_binary(ref.copy(), df.copy(), 40)

    # Remove photons labeled as surface
    print('Non-surface bathy labels')
    n_before = len(ref)
    ind = ref.loc[ref == 41].index
    ref = ref.drop(index=ind)
    df = df.drop(index=ind)
    n_after = len(ref)
    if args.verbose:
        print(f'Removed {n_before-n_after} non-bathy photons')
    score_binary(ref, df, 40)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", default=False)
    parser.add_argument('input_glob',
                        type=str,
                        help='Input training filename glob')
    args = parser.parse_args()

    main(args)
