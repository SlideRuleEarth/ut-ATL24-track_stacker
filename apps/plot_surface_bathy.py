#!/usr/bin/env python3
"""
ATL24 Bathy Track Stacker
"""

import argparse
import sys
import matplotlib.pyplot as plt
import pandas as pd


def plot(df):

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

    title = "ATL24 binary F1 Scores"

    df1 = df[df.Cls == 'surface']
    df2 = df[df.Cls == 'bathy']
    df3 = df[df.Cls == 'nonsurface']

    ax.set_title(title)
    ax.set_ylim(0.0, 1.1)
    y = ax.bar(df1.Name, df1.F1, label='surface', color='tab:green')
    ax.bar_label(y, label_type='edge')
    y = ax.bar(df3.Name, df3.F1, label='bathy', color='tab:brown')
    ax.bar_label(y, label_type='edge')
    ax.legend(loc='lower center', ncols=2)
    plt.xticks(range(len(df1.Name)), df1.Name, rotation=90)
    plt.title(title, fontsize=16)
    plt.show()


def main(args):

    # Show args
    if args.verbose:
        print(args, file=sys.stderr)

    # Get the filename
    fn = args.input_filename

    if args.verbose:
        print(f'Input result file = {fn}', file=sys.stderr)

    # Read it
    df = pd.read_csv(fn, engine='pyarrow', sep='\t')

    # Don't plot OpenOceans
    df = df[df.Name != 'openoceans']

    # Plot it
    plot(df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ATL24 feature correlations')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show verbose output')
    parser.add_argument(
        'input_filename',
        type=str,
        help='Input results text filename ')

    args = parser.parse_args()

    main(args)
