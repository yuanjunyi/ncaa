import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

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
    df = df[['Season', 'WTeamID', 'LTeamID', 'WSeedInt', 'LSeedInt']]

    df = df.merge(df_regular_season_statistics_team.rename(columns={'TeamID':'WTeamID', 'Pct':'WTeamPct', 'Margin':'WTeamMargin'}),
                  how='inner',
                  on=['Season', 'WTeamID'])\
           .merge(df_regular_season_statistics_team.rename(columns={'TeamID':'LTeamID', 'Pct':'LTeamPct', 'Margin':'LTeamMargin'}),
                  how='inner',
                  on=['Season', 'LTeamID'])
    df = df[['Season', 'WTeamID', 'LTeamID', 'WSeedInt', 'LSeedInt', 'WTeamPct', 'LTeamPct', 'WTeamMargin', 'LTeamMargin']]
    
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
    df = df[['Season', 'WTeamID', 'LTeamID', 'WSeedInt', 'LSeedInt', 'WTeamPct', 'LTeamPct', 'WTeamMargin', 'LTeamMargin', 'WConfPct', 'LConfPct', 'WConfMargin', 'LConfMargin']]
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
    df_features = df[['WSeedInt', 'LSeedInt', 'WTeamPct', 'LTeamPct', 'WTeamMargin', 'LTeamMargin', 'WConfPct', 'LConfPct', 'WConfMargin', 'LConfMargin']]
    df_features_reverse = df[['LSeedInt', 'WSeedInt', 'LTeamPct', 'WTeamPct', 'LTeamMargin', 'WTeamMargin', 'LConfPct', 'WConfPct', 'LConfMargin', 'WConfMargin']]

    M = len(df_features)
    X = np.vstack([df_features.values, df_features_reverse.values])
    y = np.hstack([np.ones(M), np.zeros(M)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    for C in np.logspace(-4, 0, 5):
        clf = LogisticRegression(C=C)
        clf.fit(X_train_scaled, y_train)
        print('C={}, log_loss={}'.format(C, log_loss(y_test, clf.predict_proba(X_test_scaled)[:, 1])))

    for C in np.logspace(-4, 0, 5):
        print()
        for gamma in np.logspace(-4, 2, 7):
            clf = SVC(kernel='rbf', C=C, gamma=gamma, probability=True)
            clf.fit(X_train_scaled, y_train)
            print('C={}, gamma={}, log_loss={}'.format(C, gamma, log_loss(y_test, clf.predict_proba(X_test_scaled)[:, 1])))