#!/usr/bin/env python3
"""
ATL24 Bathy Track Stacker
"""

import argparse
import sys
import glob
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import LocalOutlierFactor


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

    # Extract data and combine into a single dataframe
    for n, fn in enumerate(filenames):

        if args.verbose:
            print(f'Reading {n + 1} of {len(filenames)}: {fn}',
                  file=sys.stderr)

        d = pd.read_csv(fn, engine='pyarrow')

        if args.verbose:
            print(f'Read {len(d.index)} rows', file=sys.stderr)

        # Get indexes of points marked as bathy
        indexes = d.index[(d['qtrees'] == 40) |
                          (d['cshelph'] == 40) |
                          (d['medianfilter'] == 40) |
                          (d['bathypathfinder'] == 40) |
                          (d['openoceanspp'] == 40) |
                          (d['coastnet'] == 40)]

        # Get a list of photons that contain at least one bathy prediction
        p = d[['x_atc', 'geoid_corr_h']].copy().to_numpy()[indexes]

        # Apply aspect ratio
        aspect_ratio = 10
        p[0, :] /= aspect_ratio

        # Compute Local Outlier Factor
        n_neighbors = 16
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof.fit(p)

        # Get densities of bathy photons
        density = lof.negative_outlier_factor_

        # Keep only the columns we need
        d = d[['geoid_corr_h',
               'surface_h',
               'qtrees',
               'cshelph',
               'medianfilter',
               'bathypathfinder',
               'openoceanspp',
               'coastnet',
               'manual_label']]

        # Add the density
        d['density'] = density.max()
        d.loc[indexes, 'density'] = density

        if args.verbose:
            print(d.columns, file=sys.stderr)

        # Save the columns to the master dataframe
        df = pd.concat([df, d])

    if args.verbose:
        print(f'Final dataframe = {df.shape}', file=sys.stderr)
        print(df.describe(), file=sys.stderr)

    algorithms = [
        'bathypathfinder',
        'coastnet',
        'cshelph',
        'medianfilter',
        'openoceanspp',
        'qtrees',
        ]

    if args.verbose:
        x = df['manual_label'].unique()
        print(f'unique(ref): {x}', file=sys.stderr)
        for col in algorithms:
            x = df[col].unique()
            print(f'unique({col}): {x}', file=sys.stderr)

    features = algorithms.copy()
    features.append('geoid_corr_h')
    features.append('surface_h')
    features.append('density')

    if args.verbose:
        print('Features:', features, file=sys.stderr)

    x = df[df.columns.intersection(features)].copy()
    y = df['manual_label'].copy()

    # Replace 'unknown' with 'unclassified'
    y[y == 1] = 0

    # Replace 'water column' with 'unclassified'
    y[y == 45] = 0

    # Make labels consecutive
    y[y == 40] = 1
    y[y == 41] = 2

    if args.verbose:
        print('X=', file=sys.stderr)
        print(x.describe(), file=sys.stderr)
        print('Y=', file=sys.stderr)
        print(y.describe(), file=sys.stderr)

    # Create the classifier
    max_depth = 6
    clf = xgb.XGBClassifier(device='cuda', max_depth=max_depth)

    if args.verbose:
        print('Fitting...', file=sys.stderr)

    clf.fit(x, y)

    if args.verbose:
        print(f'Saving to {args.model_filename}', file=sys.stderr)

    clf.save_model(args.model_filename)

    if args.verbose:
        print('Getting predictions...', file=sys.stderr)

    p = clf.predict(x)

    if args.verbose:
        r = classification_report(y, p, digits=3)
        f1 = f1_score(y, p, average='weighted')
        ba = balanced_accuracy_score(y, p)
        print(r, file=sys.stderr)
        print(f'Weighted F1\t{f1:.3f}', file=sys.stderr)
        print(f'Balanced accuracy\t{ba:.3f}', file=sys.stderr)

        # Get feature importances
        print(f'{"col":>20}{"importance":>20}', file=sys.stderr)
        for n, col in enumerate(x.columns):
            print(f'{col:>20}{clf.feature_importances_[n]:20.5f}',
                  file=sys.stderr)

    if args.permutation_importances:
        print('Getting permutation importances...', file=sys.stderr)
        r = permutation_importance(clf,
                                   x,
                                   y,
                                   n_repeats=10,
                                   random_state=0)
        print(r, file=sys.stderr)

        for i in r.importances_mean.argsort()[::-1]:
            print(f"{x.columns[i]:<20}"
                  f"{r.importances_mean[i]:5.2f}"
                  f" +/- {r.importances_std[i]:5.2f}",
                  file=sys.stderr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ATL24 bathy track stacker')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show verbose output')
    parser.add_argument(
        '-p', '--permutation-importances', action='store_true',
        help='Compute permutation importances')
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
