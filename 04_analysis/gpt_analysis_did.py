import numpy as np
from datetime import datetime, timedelta
from utils import load_data, run_ols_and_logit

TARGET_DATE = datetime(2022, 11, 30)
TARGET_DATE_PREV = datetime(2020, 11, 30)
WINDOW_DAYS = 6 * 30

df = load_data()

start_date = TARGET_DATE - timedelta(days=WINDOW_DAYS)
end_date = TARGET_DATE + timedelta(days=WINDOW_DAYS)
start_date_prev = TARGET_DATE_PREV - timedelta(days=WINDOW_DAYS)
end_date_prev = TARGET_DATE_PREV + timedelta(days=WINDOW_DAYS)

df = df[((df['question_posted'] >= start_date) & (df['question_posted'] <= end_date)) |
        ((df['question_posted'] >= start_date_prev) & (df['question_posted'] <= end_date_prev))]

df['treatment'] = np.where(
    (df['question_posted'] >= start_date) & (df['question_posted'] <= end_date), 1, 0)

df['gpt_month'] = np.where(
    df['treatment'] == 1,
    (df['question_posted'] - TARGET_DATE).dt.days // 30,
    (df['question_posted'] - TARGET_DATE_PREV).dt.days // 30)

df['post_gpt'] = np.where(
    df['treatment'] == 1,
    df['question_posted'] > TARGET_DATE,
    df['question_posted'] > TARGET_DATE_PREV)

run_ols_and_logit(
    df,
    ols_formula='demeaned_numHelped ~ after_bounty * gpt_month * post_gpt * treatment + log_num_answers_provided * after_bounty',
    logit_formula='hasHelped ~ after_bounty * gpt_month * post_gpt * treatment + log_num_answers_provided * after_bounty',
)
