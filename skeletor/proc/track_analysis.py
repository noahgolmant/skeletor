""" Simple tools for analyzing track results after experimentation. """
import os
import pickle
import track
import pandas as pd
import argparse


def df_from_proj(proj):
    """
    Gets a flattened dataframe with all trial results for the track.Project 'proj'
    See track.Project for how to get this from a logroot directory.
    """
    results = []
    for _, trial in proj.ids.iterrows():
        res = proj.results([trial['trial_id']])
        for col in proj.ids.columns:
            # If the argument was a list (e.g. annealing schedule), have to
            # stringify it to assign it as a default val for all rows.
            if isinstance(trial[col], list):
                trial[col] = str(trial[col])
            res[col] = trial[col]
        results.append(res)
    _df = pd.concat(results)
    return _df


def proj(experimentname=None, logroot=None, s3=None,
         proj_dir=None):
    """
    Loads the track.Project object for this experiment directory.
    Gets the flattened dataframe.

    if proj_dir specified, load directly from there.
    otherwise, load from logroot/experimentname.

    loads from s3 if it can via track.
    """
    if not proj_dir:
        if experimentname:
            assert logroot, "must supply logroot if you have an experiment name"
        proj_dir = os.path.join(logroot, experimentname)
    proj = track.Project(proj_dir, s3)
    return proj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create pandas dataframes from track')
    parser.add_argument('--logroot', default='./logs')
    parser.add_argument('experimentname', default='resnet_cifar')
    args = parser.parse_args()

    _proj = proj(args.experimentname, args.logroot)
    proj_fname = os.path.join(args.logroot, args.experimentname,
                              args.experimentname + '.pkl')
    try:
        with open(proj_fname, 'wb') as f:
            pickle.dump(_proj, f)
    except Exception as e:
        print('swallowing pickle error: {}'.format(e))

