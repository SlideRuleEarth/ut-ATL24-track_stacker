#!/usr/bin/env python3
"""
ATL24 Bathy Track Stacker
"""

import argparse
import sys
import glob
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score


def main(args):

    # Show args
    if args.verbose:
        print(args, file=sys.stderr)

    # Get the filenames
    filenames = glob.glob(args.input_glob)

    if args.verbose:
        print(filenames, file=sys.stderr)
        print(f'{len(filenames)} total files', file=sys.stderr)

    # Master dataframe
    df = pd.DataFrame()

    for n, fn in enumerate(filenames):

        if args.verbose:
            print(f'Reading {n + 1} of {len(filenames)}: {fn}')

        d = pd.read_csv(fn, engine='pyarrow')

        if args.verbose:
            print(f'Read {len(df.index)} rows')

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

    for col in df.columns:
        x = df[col].unique()
        print(f'unique({col}): {x}')

    x = df[df.columns.intersection(algorithms)]
    y = df.manual_label.to_numpy()

    # Make labels consecutive
    y[y == 40] = 1
    y[y == 41] = 2

    clf = xgb.XGBClassifier(device='cuda')

    if args.verbose:
        print('Fitting...', file=sys.stderr)

    clf.fit(x, y)

    if args.verbose:
        print(f'Saving to {args.model_filename}', file=sys.stderr)

    clf.save_model(args.model_filename)

    p = clf.predict(x)
    r = classification_report(y, p, digits=3)
    print(r)
    f1 = f1_score(y, p, average='weighted')
    ba = balanced_accuracy_score(y, p)
    print(f'Weighted F1\t{f1:.3f}')
    print(f'Balanced accuracy\t{ba:.3f}')

    '''
    g = xgb.to_graphviz(clf, num_trees=1)
    g.view()

    fig, ax = plt.subplots()
    xgb.plot_tree(clf, num_trees=4, ax=ax)
    plt.show()
    '''

    # Get feature importances
    print(f'{"col":>20}{"importance":>10}')
    for n, col in enumerate(algorithms):
        print(f'{col:>20}{clf.feature_importances_[n]:10.5f}')

    print('Getting permutation importances...')
    r = permutation_importance(clf, x, y, n_repeats=3, random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{algorithms[i]:<20}"
                  f"{r.importances_mean[i]:5.2f}"
                  f" +/- {r.importances_std[i]:5.2f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ATL24 bathy track stacker')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show verbose output')
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        help='Number of training epochs')
    parser.add_argument(
        '-m', '--model-filename',
        type=str,
        help='Specify output model filename')
    parser.add_argument(
        'input_glob',
        type=str,
        help='Input training filename glob')

    args = parser.parse_args()

    main(args)
