import argparse
import pandas as pd
import sys
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score


def main(args):

    # Show args
    if args.verbose:
        print('input_filename:', args.input_filename, file=sys.stderr)
        print('model_filename:', args.model_filename, file=sys.stderr)
        print('output_filename:', args.output_filename, file=sys.stderr)

    df = pd.read_csv(args.input_filename, engine='pyarrow')

    # Replace NAN's with 0's
    df = df.fillna(0)
    # Replace 'unknown' with 'unclassified'
    df = df.replace(1.0, 0.0)
    # Replace 'water column' with 'unclassified'
    df = df.replace(45.0, 0.0)

    df = df[['ortho_h',
           'qtrees',
           'cshelph',
           'medianfilter',
           'bathypathfinder',
           'openoceans',
           'openoceanspp',
           'coastnet',
           'pointnet',
           'manual_label']]

    if args.verbose:
        print(df.columns, file=sys.stderr)

    x = df.drop('manual_label', axis=1).to_numpy()
    y = df.manual_label.to_numpy()

    # Make labels consecutive
    y[y == 40] = 1
    y[y == 41] = 2

    clf = xgb.XGBClassifier(device='cpu')
    clf.load_model(args.model_filename)

    if args.verbose:
        print('Predicting...', file=sys.stderr)

    p = clf.predict(x)
    r = classification_report(y, p, digits=3)

    if args.verbose:
        print(r, file=sys.stderr)

    f1 = f1_score(y, p, average='weighted')
    ba = balanced_accuracy_score(y, p)

    if args.verbose:
        print(f'Weighted F1\t{f1:.3f}', file=sys.stderr)
        print(f'Balanced accuracy\t{ba:.3f}', file=sys.stderr)

    # Assign predictions
    df["ensemble"] = p

    # Change labels back to APSRS
    df[df == 1] = 40
    df[df == 2] = 41

    # Save results
    df.to_csv(args.output_filename, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='classify',
        description='Classify a point cloud')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Show verbose output")
    parser.add_argument(
        'input_filename',
        help="Input filename specification")
    parser.add_argument(
        '-m', '--model-filename',
        help="Model filename specification")
    parser.add_argument(
        '-o', '--output-filename',
        help="Output filename specification")
    args = parser.parse_args()

    main(args)
