import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from datetime import datetime, timedelta

# Read the parquet file
df = pd.read_parquet('../03_processed_datasets/processed_bounty_dataset.parquet')

# Convert timestamp to datetime if it's not already in datetime format
df['BountyAmount'] = df['BountyAmount'].astype(int) / 100

# Define the target dates
target_date = datetime(2022, 11, 30)
target_date_prev = datetime(2021, 11, 30)

# Calculate the date range for the current year
start_date = target_date - timedelta(days=6*30)  # approximately 6 months before
end_date = target_date + timedelta(days=6*30)    # approximately 6 months after

# Calculate the date range for the previous year
start_date_prev = target_date_prev - timedelta(days=6*30)  # approximately 6 months before
end_date_prev = target_date_prev + timedelta(days=6*30)    # approximately 6 months after

# Filter the DataFrame to keep only rows within the date range for both years
df = df[((df['timestamp'] >= start_date) & (df['timestamp'] <= end_date) &
        (df['question_posted'] >= start_date) & (df['question_posted'] <= end_date) &
        (df['bounty_end'] >= start_date) & (df['bounty_end'] <= end_date)) |
        ((df['timestamp'] >= start_date_prev) & (df['timestamp'] <= end_date_prev) &
        (df['question_posted'] >= start_date_prev) & (df['question_posted'] <= end_date_prev) &
        (df['bounty_end'] >= start_date_prev) & (df['bounty_end'] <= end_date_prev))]

# Add the treatment variable
df['treatment'] = np.where((df['timestamp'] >= start_date) & (df['timestamp'] <= end_date), 1, 0)

# Create the gpt_weeks variable relative to the respective target dates
df['gpt_month'] = np.where(df['treatment'] == 1,
                           (df['timestamp'] - target_date).dt.days // 30,
                           (df['timestamp'] - target_date_prev).dt.days // 30)

# Create the post_gpt variable relative to the respective target dates
df['post_gpt'] = np.where(df['treatment'] == 1,
                          df['timestamp'] > target_date,
                          df['timestamp'] > target_date_prev)

# Create log_num_answers_provided variable
df['log_num_answers_provided'] = np.log(df['num_answers_provided'] + 1)

# To check the first few rows with these new columns
print(df.head())

# Define the regression formula with the treatment variable included in the interaction terms
formula = 'demeaned_numHelped ~ after_bounty * gpt_month * post_gpt * treatment + log_num_answers_provided * after_bounty'

# Fit the model using the formula and clustered standard errors
model = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[['question_id', 'user_id']].apply(tuple, axis=1)})

print(model.summary())
df['gpt_month_cat'] = df['gpt_month'].astype('category')

# Define the logistic regression formula with the treatment variable included in the interaction terms
formula = 'hasHelped ~ after_bounty * gpt_month_cat * treatment + log_num_answers_provided * after_bounty'

# Fit the logistic model using the formula and clustered standard errors
model = smf.logit(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[['question_id', 'user_id']].apply(tuple, axis=1)})

# Print the summary of the model
print(model.summary())