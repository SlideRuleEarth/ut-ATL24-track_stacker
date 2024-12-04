import argparse
import pandas as pd
import sys
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import LocalOutlierFactor

# Turn off SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def classify(df, verbose, model_filename):

    # Add a manual label column if one does not exist
    if 'manual_label' not in df.columns:
        df['manual_label'] = 0

    # Make sure manual label is an int
    df['manual_label'] = df['manual_label'].astype(int)

    # Save photon indexes
    index_ph = df['index_ph']

    # Save along track distance
    x_atc = df['x_atc']

    # Get indexes of points marked as bathy
    indexes = df.index[(df['qtrees'] == 40) |
                       (df['cshelph'] == 40) |
                       (df['medianfilter'] == 40) |
                       (df['bathypathfinder'] == 40) |
                       (df['openoceanspp'] == 40) |
                       (df['coastnet'] == 40)]

    # Get a list of photons that contain at least one bathy prediction
    p = df[['x_atc', 'geoid_corr_h']].copy().to_numpy()[indexes]

    # Apply aspect ratio
    aspect_ratio = 10
    p[0, :] /= aspect_ratio

    # Compute Local Outlier Factor
    n_neighbors = 16
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit(p)

    # Get densities of bathy photons
    density = lof.negative_outlier_factor_

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

    # Add the density
    df.loc[:, 'density'] = density.max()
    df.loc[indexes, 'density'] = density

    if verbose:
        print(df.describe(), file=sys.stderr)

    x = df.drop('manual_label', axis=1).to_numpy()
    y = df.manual_label.copy().to_numpy()

    # Replace 'unknown' with 'unclassified'
    y[y == 1] = 0

    # Replace 'water column' with 'unclassified'
    y[y == 45] = 0

    # Make labels consecutive
    y[y == 40] = 1
    y[y == 41] = 2

    clf = xgb.XGBClassifier(device='cpu')
    clf.load_model(model_filename)

    if verbose:
        print('Predicting...', file=sys.stderr)

    p = clf.predict(x)
    q = clf.predict_proba(x)[:, 1]

    if verbose:
        r = classification_report(y, p, digits=3)
        f1 = f1_score(y, p, average='weighted')
        ba = balanced_accuracy_score(y, p)

        print(r, file=sys.stderr)
        print(f'Weighted F1\t{f1:.3f}', file=sys.stderr)
        print(f'Balanced accuracy\t{ba:.3f}', file=sys.stderr)

    # Add back x_atc column for viewing
    df.loc[:, "x_atc"] = x_atc

    # Change predictions back to ASPRS
    p[p == 1] = 40
    p[p == 2] = 41

    # Assign predictions
    df.loc[:, "ensemble"] = p
    df.loc[:, "ensemble_bathy_prob"] = q

    # Add the indexes
    df.loc[:, "index_ph"] = index_ph

    if verbose:
        print(df.describe(), file=sys.stderr)

    return df


def main(args):

    # Show args
    if args.verbose:
        print('input_filename:', args.input_filename, file=sys.stderr)
        print('model_filename:', args.model_filename, file=sys.stderr)
        print('output_filename:', args.output_filename, file=sys.stderr)

    # Get the dataframe
    df = pd.read_csv(args.input_filename, engine='pyarrow')

    # Get predictions
    df = classify(df, args.verbose, args.model_filename)

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
