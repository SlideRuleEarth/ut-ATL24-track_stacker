#!/usr/bin/env python3
"""
ATL24 Bathy Track Stacker
"""

import argparse
import sys
import glob
import matplotlib.pyplot as plt
import pandas as pd


def plot_corr(title, x):
    print(x.corr())
    plt.matshow(x.corr())
    plt.xticks(range(x.shape[1]), x.columns, fontsize=14, rotation=90)
    plt.yticks(range(x.shape[1]), x.columns, fontsize=14, rotation=0)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16)
    plt.show()


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
               'geoid_corr_h',
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

    cols = [
        'manual_label',
        'bathypathfinder',
        'coastnet',
        'cshelph',
        'medianfilter',
        'openoceans',
        'pointnet',
        'qtrees',
        ]

    df = df[cols]

    # Replace NAN's with 0's
    df = df.fillna(0)
    # Replace 'unknown' with 'unclassified'
    df = df.replace(1.0, 0.0)
    # Replace 'water column' with 'unclassified'
    df = df.replace(45.0, 0.0)

    df_surface = df.replace(41.0, 0.0)
    df_bathy = df.replace(40.0, 0.0)

    plot_corr("All predictions", df)
    plot_corr("Surface predictions", df_surface)
    plot_corr("Bathy predictions", df_bathy)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ATL24 feature correlations')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show verbose output')
    parser.add_argument(
        'input_glob',
        type=str,
        help='Input training filename glob')

    args = parser.parse_args()

    main(args)
