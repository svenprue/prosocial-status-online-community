import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DATA_PATH = '../03_processed_datasets/processed_bounty_dataset.parquet'


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df['question_to_bounty_end'] = (df['bounty_end'] - df['question_posted']).dt.days
    df['question_to_answer'] = (df['timestamp'] - df['question_posted']).dt.days
    df = df[df['question_to_bounty_end'] < 30]
    df = df[df['question_to_answer'] < 30]
    df['log_num_answers_provided'] = np.log(df['num_answers_provided'] + 1)
    return df


def _cluster_groups(df):
    return df[['question_id', 'user_id']].apply(tuple, axis=1)


def run_ols_and_logit(df, ols_formula, logit_formula):
    ols = smf.ols(formula=ols_formula, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': _cluster_groups(df)})
    print(ols.summary())

    logit = smf.logit(formula=logit_formula, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': _cluster_groups(df)})
    print(logit.summary())

    return ols, logit
