#!/usr/bin/env python3
"""
ATL24 Bathy Track Stacker
"""

import argparse
import copy
import matplotlib.pyplot as plt
import pandas as pd
import sys


def avg(dfs):

    # Get the first one
    df = copy.deepcopy(dfs[0])

    cols = ['Accuracy', 'F1', 'BA', 'calF1', 'MCC', 'avg4']

    # Add the rest
    for tmp in dfs[1:]:
        for col in cols:
            df[col] = df[col] + tmp[col]

    # Get the average
    for col in cols:
        df[col] = df[col] / len(dfs)

    return df


def plot(title, df):

    # print(plt.style.available)
    # ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery',
    # '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast',
    # 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8',
    # 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark',
    # 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid',
    # 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook',
    # 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster',
    # 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white',
    # 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
    # plt.style.use('seaborn-v0_8')

    fig, ax = plt.subplots(layout='constrained')

    df1 = df[df.Cls == 'surface']
    df2 = df[df.Cls == 'bathy']
    # df3 = df[df.Cls == 'nonsurface']

    ax.set_title(title)
    ax.set_ylim(0.0, 1.1)
    y = ax.bar(df1.Name, df1.F1, label='surface', color='tab:green')
    ax.bar_label(y, label_type='edge', fmt='%0.3f')
    y = ax.bar(df2.Name, df2.F1, label='bathy', color='tab:brown')
    ax.bar_label(y, label_type='edge', fmt='%0.3f')
    ax.legend(loc='lower center', ncols=2)
    plt.xticks(range(len(df1.Name)), df1.Name, rotation=90)
    plt.title(title, fontsize=16)
    plt.show()


def main(args):

    # Show args
    if args.verbose:
        print(args, file=sys.stderr)

    # Get the filenames
    fns = args.input_filenames

    if args.verbose:
        print(f'Input result files = {fns}', file=sys.stderr)

    if len(fns) == 1:

        # Read it
        df = pd.read_csv(fns[0], engine='pyarrow', sep='\t')

    else:

        dfs = []

        # Get a list of dataframes
        for fn in fns:
            dfs.append(pd.read_csv(fn, engine='pyarrow', sep='\t'))

        # Get averages
        df = avg(dfs)

    # Don't plot these
    df = df[df.Name != 'openoceans']
    df = df[df.Name != 'coastnet']
    df = df[df.Name != 'openoceanspp']
    df = df[df.Name != 'qtrees']

    if args.verbose:
        print(df, file=sys.stderr)

    # Plot it
    if len(fns) > 1:
        title = f"ATL24 {len(fns)}-fold cross validation F1 scores"
    else:
        title = f"ATL24 F1 scores"
    plot(title, df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ATL24 feature correlations')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show verbose output')
    parser.add_argument(
        'input_filenames',
        type=str,
        nargs='+',
        help='Input results text filenames')

    args = parser.parse_args()

    main(args)
