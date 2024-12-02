#!/usr/bin/env python3
"""
ATL24 plotter
"""

import argparse
import sys
import matplotlib.pyplot as plt
import pandas as pd


def plot(title, x):

    fig, ax = plt.subplots(layout='constrained')

    ax.set_title(title)
    ax.set_ylim(0.0, 1.1)
    y = ax.bar(x.Name, x.WghtF1, label='Weighted F1')
    ax.bar_label(y, label_type='edge', fmt='%0.3f')
    y = ax.bar(x.Name, x.MacroF1, label='Macro F1')
    ax.bar_label(y, label_type='edge', fmt='%0.3f')
    ax.legend(loc='lower center', ncols=2)
    plt.xticks(range(len(x.Name)), x.Name, rotation=90)
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

    # Don't plot these
    df = df[df.Name != 'openoceans']
    df = df[df.Name != 'coastnet']
    df = df[df.Name != 'openoceanspp']
    df = df[df.Name != 'qtrees']
    df = df[df.Name != 'ensemble']

    if args.verbose:
        print(df, file=sys.stderr)

    # Plot it
    plot('ATL24 multi-class F1 scores', df)


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
