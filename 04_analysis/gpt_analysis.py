import pandas as pd
from datetime import datetime, timedelta
import statsmodels.formula.api as smf
import numpy as np

# Read the parquet file
df = pd.read_parquet('../03_processed_datasets/processed_bounty_dataset.parquet')

# Define the target date
target_date = datetime(2022, 11, 30)

# Calculate the date range
start_date = target_date - timedelta(days=6*30)  # approximately 6 months before
end_date = target_date + timedelta(days=6*30)    # approximately 6 months after

# Filter the DataFrame to keep only rows within the date range
df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date) &
        (df['question_posted'] >= start_date) & (df['question_posted'] <= end_date) &
        (df['bounty_end'] >= start_date) & (df['bounty_end'] <= end_date)]

# Create the gpt_weeks variable
df['gpt_weeks'] = (df['timestamp'] - target_date).dt.days // 7

# Create the post_gpt variable
df['post_gpt'] = df['timestamp'] > target_date

df['log_num_answers_provided'] = np.log(df['num_answers_provided'] + 1)

# To check the first few rows with these new columns
print(df.head())

# Define the regression formula
formula = 'demeaned_numHelped ~ after_bounty * gpt_weeks * post_gpt + log_num_answers_provided * after_bounty'

# Fit the model using the formula and clustered standard errors
model = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[['question_id', 'user_id']].apply(tuple, axis=1)})

print(model.summary())

# Define the logistic regression formula
formula = 'hasHelped ~ after_bounty * gpt_weeks * post_gpt + log_num_answers_provided * after_bounty'

# Fit the logistic model using the formula and clustered standard errors
model = smf.logit(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[['question_id', 'user_id']].apply(tuple, axis=1)})

# Print the summary of the model
print(model.summary())


# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Print the first row of the DataFrame to check the variables
print(df.head(5))

# Reset pandas display option if needed
pd.reset_option('display.max_columns')