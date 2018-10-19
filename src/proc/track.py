import track
import pandas as pd
import argparse


def df(logroot):
    proj = track.Project(logroot, None)
    results = []
    for _, trial in proj.ids.iterrows():
        res = proj.results([trial['trial_id']])
        for col in proj.ids.columns:
            res[col] = trial[col]
        results.append(res)
    _df = pd.concat(results)
    return _df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create pandas dataframes from track')
    parser.add_argument('--logroot', default='./logs')
    parser.add_argument('outfile', default='track.pkl')
    args = parser.parse_args()

    _df = df(args.logroot)
    _df.to_pickle(args.outfile)

