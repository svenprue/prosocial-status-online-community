from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from utils import load_data, run_ols_and_logit

TARGET_DATE = datetime(2021, 11, 30)
WINDOW_DAYS = 6 * 30

df = load_data()

start_date = TARGET_DATE - timedelta(days=WINDOW_DAYS)
end_date = TARGET_DATE + timedelta(days=WINDOW_DAYS)
df = df[(df['question_posted'] >= start_date) & (df['question_posted'] <= end_date)]

df['gpt_month'] = (df['question_posted'] - TARGET_DATE).dt.days // 30
df['gpt_month_cat'] = df['gpt_month'].astype('category')
df['post_gpt'] = df['question_posted'] > TARGET_DATE

run_ols_and_logit(
    df,
    ols_formula='demeaned_numHelped ~ after_bounty * gpt_month * post_gpt',
    logit_formula='hasHelped ~ after_bounty * gpt_month * post_gpt',
)

# Monthly observation counts
df['year_month'] = df['question_posted'].dt.to_period('M')
monthly_counts = df['year_month'].value_counts().sort_index()
monthly_unique_counts = df.groupby('year_month')['question_id'].nunique()
monthly_data = pd.DataFrame({
    'total_counts': monthly_counts,
    'unique_counts': monthly_unique_counts,
}).reset_index()
monthly_data['year_month'] = monthly_data['year_month'].astype(str)

fig = px.bar(monthly_data, x='year_month', y='total_counts',
             labels={'year_month': 'Month', 'total_counts': 'Number of Observations'},
             title='Number of Observations by Month with Unique Question IDs')
for i, row in monthly_data.iterrows():
    fig.add_annotation(x=row['year_month'], y=row['total_counts'],
                       text=str(row['unique_counts']), showarrow=False, yshift=10)
fig.update_layout(xaxis_tickangle=-45)
fig.show()
