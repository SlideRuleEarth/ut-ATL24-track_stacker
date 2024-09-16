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
    # plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(layout='constrained')

    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y')
    ax.set_axisbelow(True)

    scores = {'Avg(F1,BA,cF1,MCC)': df.avg4,
              'F1': df.F1,
              'BA': df.BA,
              'calF1': df.calF1,
              'MCC': df.MCC}
    width = 0.18
    c = 0
    n = np.arange(len(df.avg4))

    for s, d in scores.items():
        x = n + width * c
        x = x - width * len(scores) / 2.0 + width / 2.0
        y = ax.bar(x, d, width, label=s)
        ax.bar_label(y, label_type='edge')
        c += 1

    ax.legend(loc='lower center', ncols=5)
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

    df1 = df[df.Cls == 'surface']
    df2 = df[df.Cls == 'bathy']
    df3 = df[df.Cls == 'nonsurface']

    # Plot it
    plot('Sea surface scores: ' + fn, df1)
    plot('Bathy scores: ' + fn, df2)
    plot('Non-surface scores: ' + fn, df3)


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
