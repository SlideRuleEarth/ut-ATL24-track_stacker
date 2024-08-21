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


def score_all(c, a, r, d):

    print(f'{"Cls":>10}'
          f'{"Name":>20}'
          f'{"Accuracy":>10}'
          f'{"WghtF1":>10}'
          f'{"MacroF1":>10}')

    # Get the scores
    p = d[a]
    acc = accuracy_score(r, p)
    weighted_f1 = f1_score(r, p, average="weighted")
    macro_f1 = f1_score(r, p, average="macro")
    print(f'{c:>10}'
          f'{a:>20}'
          f'{acc:10.3f}'
          f'{weighted_f1:10.3f}'
          f'{macro_f1:10.3f}')


def score_binary(c, a, ref, df, pos_label):

    # Replace values
    r = ref.copy()
    d = df.copy()
    r[r != pos_label] = 0
    d[d != pos_label] = 0

    print(f'{"Cls":>10}'
          f'{"Name":>20}'
          f'{"Accuracy":>10}'
          f'{"F1":>10}'
          f'{"BA":>10}'
          f'{"calF1":>10}'
          f'{"MCC":>10}'
          f'{"avg4":>10}'
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
    print(f'{c:>10}'
          f'{a:>20}'
          f'{acc:10.3f}'
          f'{f1:10.3f}'
          f'{ba:10.3f}'
          f'{cal_f1:10.3f}'
          f'{mcc:10.3f}'
          f'{avg:10.3f}')


def main(args):

    # Get the filenames
    filenames = glob.glob(args.input_glob)

    if args.verbose:
        print(filenames, file=sys.stderr)
        print(f'{len(filenames)} total files', file=sys.stderr)

    df = pd.DataFrame()

    algorithms = [
        'qtrees',
        'cshelph',
        'medianfilter',
        'bathypathfinder',
        'openoceans',
        'openoceanspp',
        'coastnet',
        'pointnet',
        'ensemble',
        ]
    algorithms.sort()

    for n, fn in enumerate(filenames):

        if args.verbose:
            print(f'Reading {n + 1} of {len(filenames)}: {fn}', file=sys.stderr)

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
    print(f'total photons {n_total}')
    print(f'total noise photons {n_noise}')
    print(f'total bathy photons {n_bathy}')
    print(f'total surface photons {n_surface}')

    # Remove photons labeled as surface
    df2 = df[df.manual_label != 41]
    ref2 = df2['manual_label']
    if args.verbose:
        print(f'Removed {n_total-len(df2.index)} surface photons',
              file=sys.stderr)

    for a in algorithms:
        if args.verbose:
            print(f'Scoring {a}')
        score_all('all', a, ref, df)
        score_binary('surface', a, ref, df, 41)
        score_binary('bathy', a, ref, df, 40)
        score_binary('nonsurface', a, ref2, df2, 40)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", default=False)
    parser.add_argument('input_glob',
                        type=str,
                        help='Input training filename glob')
    args = parser.parse_args()

    main(args)
