""" Simple tools for analyzing track results after experimentation. """
import os
import track
import pandas as pd


def df_from_proj(track_proj):
    """
    Gets a flattened dataframe with all trial results for the track.Project
    'proj'. See track.Project for how to get this from a logroot directory.
    """
    results = []
    for _, trial in track_proj.ids.iterrows():
        res = track_proj.results([trial['trial_id']])
        for col in track_proj.ids.columns:
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
            assert logroot, "must supply logroot with experiment name"
        proj_dir = os.path.join(logroot, experimentname)
    track_proj = track.Project(proj_dir, s3)
    return track_proj
