import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def compute_regular_season_statistics_team(df_regular):
    df_regular['ScoreDiff'] = df_regular.WScore - df_regular.LScore

    grouped_winning_team = df_regular.groupby(['Season', 'WTeamID'])
    winning_statistics = grouped_winning_team.agg({'LTeamID':'count', 'ScoreDiff':'sum'})\
                                             .reset_index()\
                                             .rename(columns={'WTeamID':'TeamID',
                                                              'LTeamID':'W',
                                                              'ScoreDiff':'WAccumScore'})

    grouped_losing_team = df_regular.groupby(['Season', 'LTeamID'])
    losing_statistics = grouped_losing_team.agg({'WTeamID':'count', 'ScoreDiff':'sum'})\
                                           .reset_index()\
                                           .rename(columns={'LTeamID':'TeamID',
                                                            'WTeamID':'L',
                                                            'ScoreDiff':'LAccumScore'})

    df_statistics = pd.merge(winning_statistics, losing_statistics, how='outer', on=['Season', 'TeamID']).fillna(value=0)
    df_statistics['Pct'] = df_statistics.W / (df_statistics.W + df_statistics.L)
    df_statistics['Margin'] = (df_statistics.WAccumScore - df_statistics.LAccumScore) / (df_statistics.W + df_statistics.L)
    return df_statistics

def compute_regular_season_statistics_conference(df_regular, df_team_conference):
    df_regular['ScoreDiff'] = df_regular.WScore - df_regular.LScore

    df = df_regular.merge(df_team_conference.rename(columns={'TeamID':'WTeamID', 'ConfAbbrev':'WTeamConfAbbrev'}),
                          how='inner',
                          on=['Season', 'WTeamID'])\
                   .merge(df_team_conference.rename(columns={'TeamID':'LTeamID', 'ConfAbbrev':'LTeamConfAbbrev'}),
                          how='inner',
                          on=['Season', 'LTeamID'])
    df = df[df.WTeamConfAbbrev != df.LTeamConfAbbrev]

    grouped_winning_conf = df.groupby(['Season', 'WTeamConfAbbrev'])
    winning_statistics = grouped_winning_conf.agg({'LTeamConfAbbrev':'count', 'ScoreDiff':'sum'})\
                                             .reset_index()\
                                             .rename(columns={'WTeamConfAbbrev':'ConfAbbrev',
                                                              'LTeamConfAbbrev':'W',
                                                              'ScoreDiff':'WAccumScore'})

    grouped_losing_conf = df.groupby(['Season', 'LTeamConfAbbrev'])
    losing_statistics = grouped_losing_conf.agg({'WTeamConfAbbrev':'count', 'ScoreDiff':'sum'})\
                                           .reset_index()\
                                           .rename(columns={'LTeamConfAbbrev':'ConfAbbrev',
                                                            'WTeamConfAbbrev':'L',
                                                            'ScoreDiff':'LAccumScore'})

    df_statistics = pd.merge(winning_statistics, losing_statistics, how='outer', on=['Season', 'ConfAbbrev']).fillna(value=0)
    df_statistics['Pct'] = df_statistics.W / (df_statistics.W + df_statistics.L)
    df_statistics['Margin'] = (df_statistics.WAccumScore - df_statistics.LAccumScore) / (df_statistics.W + df_statistics.L)
    return df_statistics

def compute_features(df_ncaa_tourney,
                     df_seeds,
                     df_team_conference,
                     df_regular_season_statistics_team,
                     df_regular_season_statistics_conference):
    df_seeds['SeedInt'] = df_seeds.Seed.apply(lambda x: int(x[1:3]))
    df = df_ncaa_tourney.merge(df_seeds.rename(columns={'TeamID':'WTeamID', 'SeedInt':'WSeedInt'}),
                               how='inner',
                               on=['Season', 'WTeamID'])\
                        .merge(df_seeds.rename(columns={'TeamID':'LTeamID', 'SeedInt':'LSeedInt'}),
                               how='inner',
                               on=['Season', 'LTeamID'])
    df['SeedDiff'] = df.WSeedInt - df.LSeedInt
    df = df[['Season', 'WTeamID', 'LTeamID', 'SeedDiff']]

    df = df.merge(df_regular_season_statistics_team.rename(columns={'TeamID':'WTeamID', 'Pct':'WTeamPct', 'Margin':'WTeamMargin'}),
                  how='inner',
                  on=['Season', 'WTeamID'])\
           .merge(df_regular_season_statistics_team.rename(columns={'TeamID':'LTeamID', 'Pct':'LTeamPct', 'Margin':'LTeamMargin'}),
                  how='inner',
                  on=['Season', 'LTeamID'])
    df['TeamPctDiff'] = df.WTeamPct - df.LTeamPct
    df['TeamMarginDiff'] = df.WTeamMargin - df.LTeamMargin
    df = df[['Season', 'WTeamID', 'LTeamID', 'SeedDiff', 'TeamPctDiff', 'TeamMarginDiff']]
    
    df = df.merge(df_team_conference.rename(columns={'TeamID':'WTeamID', 'ConfAbbrev':'WTeamConfAbbrev'}),
                  how='inner',
                  on=['Season', 'WTeamID'])\
           .merge(df_team_conference.rename(columns={'TeamID':'LTeamID', 'ConfAbbrev':'LTeamConfAbbrev'}),
                  how='inner',
                  on=['Season', 'LTeamID'])\
           .merge(df_regular_season_statistics_conference.rename(columns={'ConfAbbrev':'WTeamConfAbbrev', 'Pct':'WConfPct', 'Margin':'WConfMargin'}),
                  how='inner',
                  on=['Season', 'WTeamConfAbbrev'])\
           .merge(df_regular_season_statistics_conference.rename(columns={'ConfAbbrev':'LTeamConfAbbrev', 'Pct':'LConfPct', 'Margin':'LConfMargin'}),
                  how='inner',
                  on=['Season', 'LTeamConfAbbrev'])
    df['ConfPctDiff'] = df.WConfPct - df.LConfPct
    df['ConfMarginDiff'] = df.WConfMargin - df.LConfMargin
    df = df[['Season', 'WTeamID', 'LTeamID', 'SeedDiff', 'TeamPctDiff', 'TeamMarginDiff', 'ConfPctDiff', 'ConfMarginDiff']]
    return df


if __name__ == '__main__':
    df_regular = pd.read_csv('DataFiles/RegularSeasonCompactResults.csv')
    df_team_conference = pd.read_csv('DataFiles/TeamConferences.csv')
    df_seeds = pd.read_csv('DataFiles/NCAATourneySeeds.csv')
    df_ncaa_tourney = pd.read_csv('DataFiles/NCAATourneyCompactResults.csv')
    
    df_regular_season_statistics_team = compute_regular_season_statistics_team(df_regular)
    df_regular_season_statistics_conference = compute_regular_season_statistics_conference(df_regular, df_team_conference)
    
    df = compute_features(df_ncaa_tourney,
                          df_seeds,
                          df_team_conference,
                          df_regular_season_statistics_team,
                          df_regular_season_statistics_conference)
    # df = df[df.Season < 2014]
    df_features = df[['SeedDiff', 'TeamPctDiff', 'TeamMarginDiff', 'ConfPctDiff', 'ConfMarginDiff']]
    N = len(df_features)
    X = pd.concat([df_features, -df_features])
    y = np.vstack([np.ones([N, 1]), np.zeros([N, 1])]).reshape(-1)
    scalar = StandardScaler()
    X_new = scalar.fit_transform(X)
    X_train, y_train = shuffle(X_new, y)
    X_train, y_train = shuffle(X, y)

    logreg = LogisticRegression()
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
    clf.fit(X_train, y_train)
    print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
    print()

    svc = SVC(probability=True)
    clf = GridSearchCV(svc, {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}, scoring='neg_log_loss', refit=True)
    clf.fit(X_train, y_train)
    print('Best log_loss: {:.4}, with best kernel: {}'.format(clf.best_score_, clf.best_params_['kernel']))