#!/usr/bin/env python3
"""
ATL24 Bathy Track Stacker
"""

import argparse
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(title, df):

    # print(plt.style.available)
    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots(layout='constrained')

    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y')
    ax.set_axisbelow(True)

    df = df[df.Cls != 'nonsurface']
    cls = ['surface',
           'bathy']

    width = 0.18
    c = 0
    n = np.arange(len(df.F1) / 2)

    print(df)
    for s in cls:
        df2 = df[df.Cls == s]
        x = n + width * c
        x = x - width * len(cls) / 2.0 + width / 2.0
        y = ax.bar(x, df2.F1, width, label=s)
        ax.bar_label(y, label_type='edge', fmt='%.2f')
        c += 1

    df = df[df.Cls == 'surface']
    ax.legend(loc='lower center', ncols=3)
    plt.xticks(range(len(df.Name)), df.Name, rotation=90)
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

    print(df)

    # Plot it
    plot('F1 Scores: ' + fn, df)


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
