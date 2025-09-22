
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit



schedule = nfl.import_schedules(list(range(2020, 2024)))
# print(schedule.columns.tolist())


pbp = nfl.import_pbp_data(list(range(2020, 2024)))
# print(pbp.head())
# print(pbp.columns.tolist())

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else np.nan

def compute_team_stats(group):
    return pd.Series(
        {
            'avg_epa': group['epa'].mean(),
            'avg_air_epa': group.loc[group['pass_attempt'] == 1, 'air_epa'].mean(),
            'avg_yac_epa': group.loc[group['pass_attempt'] == 1, 'yac_epa'].mean(),
            'avg_comp_air_epa': group.loc[group['pass_attempt'] == 1, 'comp_air_epa'].mean(),
            'avg_comp_yac_epa': group.loc[group['pass_attempt'] == 1, 'comp_yac_epa'].mean(),
            'avg_rush_epa': group.loc[group['rush_attempt'] == 1, 'epa'].mean(),
            'avg_pass_epa': group.loc[group['pass_attempt'] == 1, 'epa'].mean(),
            'avg_yds_gained': group['yards_gained'].mean(),
            'avg_passing_yds': group['passing_yards'].sum() / group['pass_attempt'].sum(),
            'avg_rushing_yds': group['rushing_yards'].sum() / group['rush_attempt'].sum(),
            'avg_air_yards': group['air_yards'].sum() / group['pass_attempt'].sum(),
            'pct_shotgun': group['shotgun'].mean(),
            'pct_qb_dropback': group['qb_dropback'].mean(),
            'avg_yards_after_catch': group['yards_after_catch'].mean(),
            'field_goal_conv_rate': safe_divide((group['field_goal_result'] == 'made').sum(),
                                                group['field_goal_attempt'].sum()),
            'avg_kick_distance': group['kick_distance'].mean(),
            'extra_point_conv_rate': safe_divide((group['extra_point_result'] == 'good').sum(),
                                                 group['extra_point_attempt'].sum()),
            'two_point_conv_rate': safe_divide((group['two_point_conv_result'] == 'success').sum(),
                                               group['two_point_attempt'].sum()),
            'third_down_conv_rate': safe_divide(group['third_down_converted'].sum(), (
                        group['third_down_converted'].sum() + group['third_down_failed'].sum())),
            'fourth_down_conv_rate': safe_divide(group['fourth_down_converted'].sum(), (
                        group['fourth_down_converted'].sum() + group['fourth_down_failed'].sum())),
            'fumble_rate': group['fumble'].mean(),
            'sack_rate': group['sack'].mean(),
            'tackled_for_loss_rate': group['tackled_for_loss'].mean(),
            'qb_hit_rate': group['qb_hit'].mean(),
            'pass_completion_rate': group['complete_pass'].sum() / group['pass_attempt'].sum(),
            'interception_rate': group['interception'].sum() / group['pass_attempt'].sum(),
            'touchdown_rate': group['touchdown'].mean(),
            'pass_touchdown_rate': group['pass_touchdown'].sum() / group['pass_attempt'].sum(),
            'rush_touchdown_rate': group['rush_touchdown'].sum() / group['rush_attempt'].sum()

        }
    )

# Apply the function to grouped data
offense_team_stats = (
    pbp.groupby(['posteam', 'season', 'week'])
        .apply(compute_team_stats)
        .reset_index()
        .rename(columns={'posteam': 'team'})
        .sort_values(['team', 'season', 'week'])
)

prefix = 'off_'
offense_team_stats = offense_team_stats.rename(columns={
    col: f'{prefix}{col}' for col in offense_team_stats.columns if col not in ['team', 'season', 'week']
})


defense_team_stats = (
    pbp.groupby(['defteam', 'season', 'week'])
        .apply(compute_team_stats)
        .reset_index()
        .rename(columns={'defteam': 'team'})
        .sort_values(['team', 'season', 'week'])
)
prefix = 'def_'
defense_team_stats = defense_team_stats.rename(columns={
    col: f'{prefix}{col}' for col in defense_team_stats.columns if col not in ['team', 'season', 'week']
})

#print(offense_team_stats.head())
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(offense_team_stats)

home = schedule[['season', 'week', 'home_team']].copy()
home['team'] = home['home_team']
home['is_home'] = True

away = schedule[['season', 'week', 'away_team']].copy()
away['team'] = away['away_team']
away['is_home'] = False

schedule_stats_df = pd.concat([home, away], ignore_index=True)
schedule_stats_df = schedule_stats_df.sort_values(['team', 'season', 'week'])
schedule_stats_df = schedule_stats_df.drop(columns=['home_team', 'away_team'])

schedule_stats_df = schedule_stats_df.merge(
    offense_team_stats,
    on=['season', 'week', 'team'],
    how='left'
)

schedule_stats_df = schedule_stats_df.merge(
    defense_team_stats,
    on=['season', 'week', 'team'],
    how='left'
)


# print(schedule_stats_df.columns.tolist())
# print(schedule_stats_df.head(5))

# cols for moving average
stat_cols = ['off_avg_epa', 'off_avg_air_epa', 'off_avg_yac_epa', 'off_avg_comp_air_epa', 'off_avg_comp_yac_epa', 'off_avg_rush_epa', 'off_avg_pass_epa', 'off_avg_yds_gained', 'off_avg_passing_yds', 'off_avg_rushing_yds', 'off_avg_air_yards', 'off_pct_shotgun', 'off_pct_qb_dropback', 'off_avg_yards_after_catch', 'off_field_goal_conv_rate', 'off_avg_kick_distance', 'off_extra_point_conv_rate', 'off_two_point_conv_rate', 'off_third_down_conv_rate', 'off_fourth_down_conv_rate', 'off_fumble_rate', 'off_sack_rate', 'off_tackled_for_loss_rate', 'off_qb_hit_rate', 'off_pass_completion_rate', 'off_interception_rate', 'off_touchdown_rate', 'off_pass_touchdown_rate', 'off_rush_touchdown_rate', 'def_avg_epa', 'def_avg_air_epa', 'def_avg_yac_epa', 'def_avg_comp_air_epa', 'def_avg_comp_yac_epa', 'def_avg_rush_epa', 'def_avg_pass_epa', 'def_avg_yds_gained', 'def_avg_passing_yds', 'def_avg_rushing_yds', 'def_avg_air_yards', 'def_pct_shotgun', 'def_pct_qb_dropback', 'def_avg_yards_after_catch', 'def_field_goal_conv_rate', 'def_avg_kick_distance', 'def_extra_point_conv_rate', 'def_two_point_conv_rate', 'def_third_down_conv_rate', 'def_fourth_down_conv_rate', 'def_fumble_rate', 'def_sack_rate', 'def_tackled_for_loss_rate', 'def_qb_hit_rate', 'def_pass_completion_rate', 'def_interception_rate', 'def_touchdown_rate', 'def_pass_touchdown_rate', 'def_rush_touchdown_rate']

running_avgs = (
    schedule_stats_df
    .groupby(['team', 'season'])[stat_cols]
    .expanding()
    .mean()
    .reset_index()
)

running_avgs = running_avgs.drop(columns=['level_2'])

moving_average_df = pd.DataFrame()
moving_average_df['season'] = schedule_stats_df['season']
moving_average_df['week'] = schedule_stats_df['week']
moving_average_df['team'] = schedule_stats_df['team']
moving_average_df['is_home'] = schedule_stats_df['is_home']

for col in stat_cols:
    moving_average_df[f'{col}_ma'] = running_avgs[col].groupby(
        schedule_stats_df['team']
    ).shift(1)

# print(moving_average_df.head(5))

# copy schedule df
score_regression_df = schedule[['season', 'week', 'home_team', 'away_team', 'result', 'spread_line']].copy()

# merge home stats
score_regression_df = score_regression_df.merge(moving_average_df,
              left_on=['season', 'week', 'home_team'],
              right_on=['season', 'week', 'team'])

# merge away stats
score_regression_df = score_regression_df.merge(moving_average_df,
              left_on=['season', 'week', 'away_team'],
              right_on=['season', 'week', 'team'],
              suffixes=('_home','_away'))

# print(score_regression_df.columns.tolist())

score_regression_df = score_regression_df.drop(columns=['is_home_home', 'is_home_away', 'team_home', 'team_away'])

score_regression_df = score_regression_df[score_regression_df['week'] != 1].copy()

# print(score_regression_df.columns.tolist())
# print(score_regression_df.iloc[20].tolist())

feature_cols = [col for col in score_regression_df.columns if '_home' in col or '_away' in col]

# X = df[]
# y = 1 if df['result'] > 0, else 0

# = LogisticRegression().fit(X, y)

'''
y = score_regression_df['result']
X = score_regression_df[feature_cols]
X = X.fillna(X.mean())

model = LinearRegression()
model.fit(X, y)

# print(model.intercept_)
# print(model.coef_)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"R²: {r2}")
print(f"RMSE: {rmse}")
'''

'''
X_train, X_test, y_train, y_test = train_test_split(score_regression_df[feature_cols], score_regression_df['result'], test_size=0.2, random_state=42)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

model = LinearRegression()
model.fit(X_train, y_train)

# print(model.intercept_)
# print(model.coef_)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Train/Test R²: {r2}')
print(f'Train/Test RMSE: {rmse}')

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
'''

# Testing on 2023, Training Before
X_train = score_regression_df[score_regression_df['season'] < 2023][feature_cols]
X_test = score_regression_df[score_regression_df['season'] == 2023][feature_cols]

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

y_train = score_regression_df[score_regression_df['season'] < 2023]['result']
y_test = score_regression_df[score_regression_df['season'] == 2023]['result']

model = LinearRegression()
model.fit(X_train, y_train)

# print(model.intercept_)
# print(model.coef_)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('Testing on 2023')
print(f'Train/Test R²: {r2}')
print(f'Train/Test RMSE: {rmse}')

tscv = TimeSeriesSplit()

lasso = LassoCV(cv=tscv)
lasso.fit(X_train, y_train)
print("Best alpha:", lasso.alpha_)

coef_df = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lasso.coef_
})

# Show only non-zero coefficients (used by the model)
lasso_used_features = coef_df[coef_df['coefficient'] != 0]
print(lasso_used_features.sort_values(by='coefficient', ascending=False))

lasso_dropped_features = coef_df[coef_df['coefficient'] == 0]
print(lasso_dropped_features)

coef_df.sort_values(by='coefficient', key=abs, ascending=False).plot.bar(x='feature', y='coefficient')

y_pred_lasso = lasso.predict(X_test)

r2 = r2_score(y_test, y_pred_lasso)

mse = mean_squared_error(y_test, y_pred_lasso)
rmse = np.sqrt(mse)

print('LASSO')
print('Testing on 2024')
print(f'Train/Test R²: {r2}')
print(f'Train/Test RMSE: {rmse}')

correct_prediction = (
    ((y_pred_lasso > score_regression_df[score_regression_df['season'] == 2023]['spread_line']) &
     (score_regression_df[score_regression_df['season'] == 2023]['result'] > score_regression_df[score_regression_df['season'] == 2023]['spread_line'])) |
    ((y_pred_lasso <= score_regression_df[score_regression_df['season'] == 2023]['spread_line']) &
     (score_regression_df[score_regression_df['season'] == 2023]['result'] <= score_regression_df[score_regression_df['season'] == 2023]['spread_line']))
)

accuracy = correct_prediction.mean()
print(f'Prediction Accuracy: {accuracy}')

selected_columns = lasso_used_features['feature'].tolist()

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_vif = X_train[selected_columns]

vif = pd.DataFrame()
vif["feature"] = X_train_vif.columns
vif["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
print(vif)


# ------------------------------------------------------------------------------

years = sorted(score_regression_df['season'].unique())

results = []

for i in range(len(years) - 1):
    train_years = years[:i + 1]
    test_year = years[i + 1]

    train_df = score_regression_df[score_regression_df['season'].isin(train_years)]
    test_df = score_regression_df[score_regression_df['season'] == test_year]

    X_train = train_df[selected_columns]
    X_test = test_df[selected_columns]
    y_train = train_df['result']
    y_test = test_df['result']

    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results.append({
        'train_years': train_years,
        'test_year': test_year,
        'rmse': rmse
    })


results_df = pd.DataFrame(results)
print(results_df)

# ------------------------------------------------------------------------------







'''
# nfl.see_weekly_cols()
weekly = nfl.import_weekly_data([2022, 2023])

# print(weekly.head())
# print(weekly.columns.tolist())


seasonal = nfl.import_seasonal_data([2022, 2023])

# print(seasonal.head())
# print(seasonal.columns.tolist())
'''