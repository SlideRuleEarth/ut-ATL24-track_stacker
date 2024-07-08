#!/usr/bin/env python3
"""
ATL24 Bathy Track Stacker
"""

import argparse
import sys
import glob
import pandas as pd


def main(args):

    # Show args
    if args.verbose:
        print(args, file=sys.stderr)

    # Get the filenames
    filenames = glob.glob(args.input_glob)

    if args.verbose:
        print(filenames, file=sys.stderr)
        print(f'{len(filenames)} total files', file=sys.stderr)

    for n, fn in enumerate(filenames):

        if args.verbose:
            print(f'Reading {n + 1} of {len(filenames)}: {fn}')

        df = pd.read_csv(fn, engine='pyarrow')

        if args.verbose:
            print(f'Read {len(df.index)} rows')

        # qtrees,cshelph,medianfilter


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
