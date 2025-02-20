import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Read the parquet file
df = pd.read_parquet('../03_processed_datasets/processed_bounty_dataset.parquet')

# Print all column names
print(df.columns)
df['question_to_bounty_end'] = (df['bounty_end'] - df['question_posted']).dt.days
df['question_to_answer'] = (df['timestamp'] - df['question_posted']).dt.days
df = df[df['question_to_bounty_end'] < 30]
df = df[df['question_to_answer'] < 30]

df['BountyAmount'] = df['BountyAmount'].astype(int) / 100

# Count the occurrences of each user_id
user_counts = df['user_id'].value_counts() / 2

# Initialize bins
bins = [1, 2, 3, 4, 5, 10, user_counts.max()]
labels = ['1', '2', '3', '4', '5-10', '>10']

# Categorize the user counts into bins
user_bins = pd.cut(user_counts, bins=bins, labels=labels, right=False, include_lowest=True)

# Create a distribution of the bins
distribution = user_bins.value_counts(sort=False)

print(distribution)


# Select the columns of interest
columns_of_interest = ['numHelped', 'num_questions_asked', 'num_answers_provided', 'after_bounty']

# Filter the dataframe to keep only these columns
df_filtered = df[columns_of_interest]

# Group by 'after_bounty' and describe the statistics
description_table = df_filtered.groupby('after_bounty').describe()

print(description_table)
df['log_num_answers_provided'] = np.log(df['num_answers_provided'] + 1)
df['log_num_questions_asked'] = np.log(df['num_questions_asked'] + 1)

# Define the regression formula
formula = 'demeaned_numHelped ~ after_bounty * log_num_answers_provided + after_bounty * log_num_questions_asked'

# Fit the model using the formula and clustered standard errors
model = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[['question_id', 'user_id']].apply(tuple, axis=1)})

# Print the summary of the model
print(model.summary())

# Define the logistic regression formula
formula = 'hasHelped ~ after_bounty * log_num_answers_provided + after_bounty * log_num_questions_asked'

# Fit the logistic model using the formula and clustered standard errors
model = smf.logit(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[['question_id', 'user_id']].apply(tuple, axis=1)})

# Print the summary of the model
print(model.summary())

filtered_df = df[(df['num_questions_asked'].isin([0, 1])) & (df['num_answers_provided'] == 0)]

def categorize_experience_receiving(row):
    if row['num_questions_asked'] == 0:
        return 'no help seeked'
    elif row['num_questions_asked'] > 0 and row['num_accepted_answers_received'] == 0:
        return 'help seeked'
    elif row['num_questions_asked'] > 0 and row['num_accepted_answers_received'] > 0:
        return 'help received'

filtered_df['initial_experience_receiving'] = filtered_df.apply(categorize_experience_receiving, axis=1)

# Define the desired order of categories for initial_experience_receiving
receiving_categories_order = [
    'no help seeked',
    'help seeked',
    'help received'
]

filtered_df['initial_experience_receiving'] = pd.Categorical(filtered_df['initial_experience_receiving'], categories=receiving_categories_order, ordered=True)

# Define the logistic regression formula
formula = 'hasHelped ~ after_bounty * initial_experience_receiving + BountyAmount * after_bounty'

# Fit the logistic model using the formula and clustered standard errors
model = smf.logit(formula=formula, data=filtered_df).fit(cov_type='cluster', cov_kwds={'groups': filtered_df[['question_id', 'user_id']].apply(tuple, axis=1)})

# Print the summary of the model
print(model.summary())

filtered_df = df[(df['num_questions_asked'] > 0) & (df['num_answers_provided'].isin([0, 1]))]

def categorize_experience_giving(row):
    if row['num_answers_provided'] == 0:
        return 'no help attempted'
    elif row['num_answers_provided'] == 1 and row['num_accepted_answers_provided'] == 0:
        return 'help attempted'
    elif row['num_accepted_answers_provided'] == 1:
        return 'helped'

# Apply the categorization functions
filtered_df['initial_experience_giving'] = filtered_df.apply(categorize_experience_giving, axis=1)

# Define the desired order of categories for initial_experience_giving
giving_categories_order = [
    'no help attempted',
    'help attempted',
    'helped'
]

filtered_df['initial_experience_giving'] = pd.Categorical(filtered_df['initial_experience_giving'], categories=giving_categories_order, ordered=True)

# Define the logistic regression formula
formula = 'hasHelped ~ after_bounty * initial_experience_giving + BountyAmount * after_bounty'

# Fit the logistic model using the formula and clustered standard errors
model = smf.logit(formula=formula, data=filtered_df).fit(cov_type='cluster', cov_kwds={'groups': filtered_df[['question_id', 'user_id']].apply(tuple, axis=1)})

# Print the summary of the model
print(model.summary())
