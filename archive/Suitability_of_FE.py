import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import numpy as np
from scipy import stats
import gc


def f_test_individual_effects_large_n(df, pooled_formula, fe_formula,
                                      original_dv, demeaned_dv, cluster_var='userId'):
    """
    F-test for individual effects with large number of groups
    Uses within-transformed data to avoid fitting millions of dummies

    Args:
        pooled_formula: Formula using original DV
        fe_formula: Formula using demeaned DV
        original_dv: Name of original dependent variable
        demeaned_dv: Name of demeaned dependent variable
    """
    print("=== F-Test for Individual Effects (Large N approach) ===")
    print("H0: No individual effects (pooled OLS appropriate)")
    print("H1: Individual effects exist (FE needed)\n")

    # 1. Fit Pooled OLS using original data
    print(f"Pooled OLS formula: {pooled_formula}")
    pooled_model = smf.ols(formula=pooled_formula, data=df).fit()
    rss_pooled = pooled_model.ssr
    n_obs = pooled_model.nobs
    k_pooled = len(pooled_model.params)

    # 2. Fit FE model using within-transformed (demeaned) data
    print(f"Fixed Effects formula: {fe_formula}")
    fe_model = smf.ols(formula=fe_formula, data=df).fit()
    rss_fe = fe_model.ssr
    k_fe = len(fe_model.params)

    # 3. Calculate F-statistic
    n_individuals = df[cluster_var].nunique()

    # Degrees of freedom
    df_fe_total = n_obs - k_fe - n_individuals + 1  # Adjust for absorbed individual effects
    df_pooled_total = n_obs - k_pooled
    df_diff = n_individuals - 1  # Number of individual effects being tested

    # F-statistic
    f_stat = ((rss_pooled - rss_fe) / df_diff) / (rss_fe / df_fe_total)
    p_value = 1 - stats.f.cdf(f_stat, df_diff, df_fe_total)

    print(f"\nResults:")
    print(f"Number of individuals: {n_individuals:,}")
    print(f"Number of observations: {n_obs:,}")
    print(f"Testing {df_diff:,} individual effects")
    print(f"Pooled OLS RSS: {rss_pooled:,.2f}")
    print(f"Fixed Effects RSS: {rss_fe:,.2f}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"Degrees of freedom: ({df_diff:,}, {df_fe_total:,})")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.001:
        print("*** Reject H0 at p < 0.001: Individual effects are significant (FE needed)")
    elif p_value < 0.01:
        print("** Reject H0 at p < 0.01: Individual effects are significant (FE needed)")
    elif p_value < 0.05:
        print("* Reject H0 at p < 0.05: Individual effects are significant (FE needed)")
    else:
        print("Fail to reject H0: No evidence of individual effects")

    return f_stat, p_value, pooled_model, fe_model


def hausman_test_large_n(df, fe_formula, re_formula, cluster_var='userId', max_groups=None):
    """
    Hausman Test for large datasets
    Optionally subsample groups if RE model is too slow
    """
    print("\n=== Hausman Test ===")
    print("H0: Random effects model is consistent")
    print("H1: Fixed effects model is necessary\n")

    # For very large datasets, consider subsampling for RE model
    if max_groups is not None:
        n_groups = df[cluster_var].nunique()
        if n_groups > max_groups:
            print(f"Subsampling {max_groups:,} groups out of {n_groups:,} for computational feasibility")
            sampled_groups = df[cluster_var].drop_duplicates().sample(max_groups, random_state=42)
            df_sample = df[df[cluster_var].isin(sampled_groups)]
        else:
            df_sample = df
    else:
        df_sample = df

    print(f"Using {df_sample[cluster_var].nunique():,} groups for Hausman test")

    # 1. Fit Fixed Effects model
    print(f"Fixed Effects formula: {fe_formula}")
    fe_model = smf.ols(formula=fe_formula, data=df_sample).fit()

    # 2. Fit Random Effects model
    print(f"Random Effects formula: {re_formula}")
    try:
        re_model = MixedLM.from_formula(re_formula, data=df_sample, groups=df_sample[cluster_var]).fit()

        # Rest of Hausman test logic...
        fe_params = fe_model.params
        re_params = re_model.params

        # Find common parameters
        common_params = []
        for param in re_params.index:
            if param in fe_params.index and param != 'Intercept':
                common_params.append(param)

        if len(common_params) == 0:
            print("No common parameters found for comparison.")
            return None, None, fe_model, re_model

        fe_coefs = fe_params[common_params].values
        re_coefs = re_params[common_params].values

        fe_cov = fe_model.cov_params().loc[common_params, common_params].values
        re_cov = re_model.cov_params().loc[common_params, common_params].values

        # Calculate Hausman statistic
        diff = fe_coefs - re_coefs
        cov_diff = fe_cov - re_cov

        try:
            # Add regularization for numerical stability
            eigenvals = np.linalg.eigvals(cov_diff)
            if np.min(eigenvals) < 1e-8:
                cov_diff += np.eye(len(cov_diff)) * 1e-8

            hausman_stat = diff.T @ np.linalg.inv(cov_diff) @ diff
            df_hausman = len(common_params)
            p_value_hausman = 1 - stats.chi2.cdf(hausman_stat, df_hausman)

            print(f"Common parameters tested: {common_params}")
            print(f"Hausman statistic: {hausman_stat:.4f}")
            print(f"Degrees of freedom: {df_hausman}")
            print(f"P-value: {p_value_hausman:.6f}")

            if p_value_hausman < 0.001:
                print("*** Reject H0 at p < 0.001: Fixed effects model is necessary")
            elif p_value_hausman < 0.01:
                print("** Reject H0 at p < 0.01: Fixed effects model is necessary")
            elif p_value_hausman < 0.05:
                print("* Reject H0 at p < 0.05: Fixed effects model is necessary")
            else:
                print("Fail to reject H0: Random effects model is consistent")

            return hausman_stat, p_value_hausman, fe_model, re_model

        except Exception as e:
            print(f"Error computing Hausman statistic: {e}")
            return None, None, fe_model, re_model

    except Exception as e:
        print(f"Error fitting random effects model: {e}")
        return None, None, fe_model, None


# Usage for your 5M user dataset:
print("Testing Fixed Effects vs Random Effects (Large Dataset)")
print("=" * 60)

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
import numpy as np
import os
from tqdm import tqdm
import gc
from numba import njit
import duckdb
# Basic Models with just phase interaction
import sys
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from IPython.display import HTML
import numpy as np
import pandas as pd
from stargazer.stargazer import Stargazer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Define required columns based on usage in the notebook
required_columns = [
    'numHelped',
    'phase',
    'hasAnswer',
    'userId',
    'year',
    'userFeNumHelped',
    'logNumHelpProvidedAT',
    'logNumQuestionsAskedAT',
]

df = pd.read_parquet('../data/study_datasets/question_centered_model_7d.parquet',
                    columns=required_columns)

# Define formulas - NO C(userId) for the F-test!
pooled_formula = "numHelped ~ C(phase)*C(hasAnswer)*logNumHelpProvidedAT + logNumQuestionsAskedAT"
fe_formula = "userFeNumHelped ~ C(phase)*C(hasAnswer)*logNumHelpProvidedAT + logNumQuestionsAskedAT"
re_formula = "numHelped ~ C(phase)*C(hasAnswer)*logNumHelpProvidedAT + logNumQuestionsAskedAT"

# Check if you have the original variable
if 'numHelped' not in df.columns:
    print("Original 'numHelped' not found. Please specify the original variable name.")
    print("Available columns:", [col for col in df.columns if 'helped' in col.lower()])
    # You may need to adjust these variable names

# 1. F-test
f_stat, f_pval, pooled_model, fe_model = f_test_individual_effects_large_n(
    df, pooled_formula, fe_formula,
    original_dv='numHelped', demeaned_dv='userFeNumHelped', cluster_var='userId'
)

# 2. Hausman test (subsample if needed)
hausman_stat, hausman_pval, fe_model2, re_model = hausman_test_large_n(
    df, fe_formula, re_formula, cluster_var='userId',
    max_groups=10000  # Subsample to 10K users for computational feasibility
)