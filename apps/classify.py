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
    # Make sure manual label is an int
    df[['manual_label']] = df[['manual_label']].astype(int)

    # Save photon indexes
    index_ph = df[['index_ph']]

    # Save along track distance
    x_atc = df[['x_atc']]

    # Keep only the columns we need
    df = df[['geoid_corr_h',
             'surface_h',
             'qtrees',
             'cshelph',
             'medianfilter',
             'bathypathfinder',
             'openoceanspp',
             'coastnet',
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
    q = clf.predict_proba(x)[:, 1]
    r = classification_report(y, p, digits=3)

    if args.verbose:
        print(r, file=sys.stderr)

    f1 = f1_score(y, p, average='weighted')
    ba = balanced_accuracy_score(y, p)

    if args.verbose:
        print(f'Weighted F1\t{f1:.3f}', file=sys.stderr)
        print(f'Balanced accuracy\t{ba:.3f}', file=sys.stderr)

    # Add back x_atc column for viewing
    df["x_atc"] = x_atc

    # Assign predictions
    df["ensemble"] = p
    df["ensemble_bathy_prob"] = q

    # Change labels back to APSRS
    df[df == 1] = 40
    df[df == 2] = 41

    # Add the indexes
    df["index_ph"] = index_ph

    # Save results
    df.to_csv(args.output_filename, index=False, float_format='%.7f')


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
