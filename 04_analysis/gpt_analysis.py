import pandas as pd
from datetime import datetime, timedelta
import statsmodels.formula.api as smf
import numpy as np
import plotly.express as px

# Read the parquet file
df = pd.read_parquet('../03_processed_datasets/processed_bounty_dataset.parquet')
# Set pandas to display all columns
pd.set_option('display.max_columns', None)
filtered_df = df[df['question_id'] == 72484611]
print(filtered_df)
pd.reset_option('display.max_columns')

# Define the target date
target_date = datetime(2021, 11, 30)
df['question_to_bounty_end'] = (df['bounty_end'] - df['question_posted']).dt.days
df['question_to_answer'] = (df['timestamp'] - df['question_posted']).dt.days
df = df[df['question_to_bounty_end'] < 30]
df = df[df['question_to_answer'] < 30]

# Calculate the date range
start_date = target_date - timedelta(days=6*30)  # approximately 6 months before
end_date = target_date + timedelta(days=6*30)    # approximately 6 months after

# Filter the DataFrame to keep only rows within the date range
df = df[(df['question_posted'] >= start_date) & (df['question_posted'] <= end_date)]

# Create the gpt_weeks variable
df['gpt_month'] = (df['question_posted'] - target_date).dt.days // 30
df['gpt_month_cat'] = df['gpt_month'].astype('category')

# Create the post_gpt variable
df['post_gpt'] = df['question_posted'] > target_date

df['log_num_answers_provided'] = np.log(df['num_answers_provided'] + 1)

# To check the first few rows with these new columns
print(df.head())

# Define the regression formula
formula = 'demeaned_numHelped ~ after_bounty * gpt_month * post_gpt'

# Fit the model using the formula and clustered standard errors
model = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[['question_id', 'user_id']].apply(tuple, axis=1)})

print(model.summary())

# Define the logistic regression formula
formula = 'hasHelped ~ after_bounty * gpt_month * post_gpt'

# Fit the logistic model using the formula and clustered standard errors
model = smf.logit(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[['question_id', 'user_id']].apply(tuple, axis=1)})

# Print the summary of the model
print(model.summary())


import pandas as pd
import plotly.express as px

# Assuming df is your DataFrame

# Convert the question_posted column to datetime format
df['question_posted'] = pd.to_datetime(df['question_posted'])

# Create a new column for year and month
df['year_month'] = df['question_posted'].dt.to_period('M')

# Group by year_month and count the number of observations and unique question_ids
monthly_counts = df['year_month'].value_counts().sort_index()
monthly_unique_counts = df.groupby('year_month')['question_id'].nunique()

# Create a combined DataFrame for plotting
monthly_data = pd.DataFrame({
    'total_counts': monthly_counts,
    'unique_counts': monthly_unique_counts
}).reset_index()

# Convert the 'year_month' to string for plotly compatibility
monthly_data['year_month'] = monthly_data['year_month'].astype(str)

# Plot using plotly
fig = px.bar(monthly_data, x='year_month', y='total_counts',
             labels={'year_month': 'Month', 'total_counts': 'Number of Observations'},
             title='Number of Observations by Month with Unique Question IDs')

# Add annotations for unique counts
for i in range(len(monthly_data)):
    fig.add_annotation(x=monthly_data['year_month'][i],
                       y=monthly_data['total_counts'][i],
                       text=str(monthly_data['unique_counts'][i]),
                       showarrow=False,
                       yshift=10)

fig.update_layout(xaxis_tickangle=-45)
fig.show()