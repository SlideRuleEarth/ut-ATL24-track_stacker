#!/usr/bin/env python3
"""
ATL24 Bathy Track Stacker
"""

import argparse
import sys
import matplotlib.pyplot as plt
import pandas as pd


def plot(fn, title, x):

    fig, ax = plt.subplots(layout='constrained')

    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    print(x.columns)
    y=ax.bar(x.Name, x.WghtF1, label='Weighted F1')
    ax.bar_label(y, label_type='edge')
    y=ax.bar(x.Name, x.MacroF1, label='Macro F1')
    ax.bar_label(y, label_type='edge')
    ax.legend(loc='lower center', ncols=2)
    plt.xticks(range(len(x.Name)), x.Name, rotation=90)
    plt.title(title, fontsize=16)
    plt.show()
    #plt.savefig(fn)


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
    print(df)

    # Don't plot OpenOceans
    df = df[df.Name != 'openoceans']

    # Plot it
    plot(fn, 'Multi-class scores: ' + fn, df)


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
