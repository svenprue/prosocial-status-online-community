import ast
import pickle
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import multiprocessing
import gc
from tqdm import tqdm

# --- SSH/Headless Configuration ---
matplotlib.use('Agg')

_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_DIR = _SCRIPT_DIR.parent
DATA_PATH = _BASE_DIR / "data" / "study_datasets" / "question_centered_model_7d_processed.parquet"
# Output is input for preprocessing/create_matched_event_histories.py and calculate_matched_questions_helping.py
OUTPUT_MATCHED_PATH = _BASE_DIR / "data" / "input" / "matched_questions.parquet"
TAG_ACCEPT_SHARES_CACHE_PATH = _BASE_DIR / "data" / "study_datasets" / "question_centered_model_7d_tag_accept_shares.pkl"
TAG_DICTIONARY_PATH = _BASE_DIR / "data" / "input" / "question_centered_model_7d_all_questions_tags.parquet"
BALANCE_TEX_PATH = _SCRIPT_DIR / "balance_check.tex"
PLOT_COMMON_SUPPORT = _SCRIPT_DIR / "common_support.pdf"
PLOT_LOVE = _SCRIPT_DIR / "love_plot.pdf"

TOP_K_TAGS = 50  # used only for tie-break / reference; PSM uses tag accept-share summary
MAX_NEIGHBORS_FOR_TIEBREAK = 100  # number of nearest controls to consider for Jaccard tie-break
# Parallel matching: strata per batch (each worker gets this many strata per pickled job to reduce overhead)
PSM_STRATA_BATCH_SIZE = 200

USE_TAG_ACCEPT_SHARE = True   # if True, use tag_accept_share_avg (and optionally _max); else one-hot
TAG_ACCEPT_SHARE_AVG_ONLY = True  # if True, only z_tag_accept_share_avg; else also z_tag_accept_share_max
TAG_ACCEPT_SHARE_PER_YEAR = True
MIN_TAG_COUNT_FOR_SHARE = 100
MIN_QUESTIONS_PER_STRATUM = 300  # exclude year-tag strata with fewer questions from matching

REQUIRED_COLUMNS = [
    'numHelped', 'phase', 'hasAnswer', 'userId', 'year',
    'timeSinceFirstActivityDays', 'numHelpProvidedAT',
    'numQuestionsAskedAT', 'responseTimeHours', 'hasSelfAnswer', 'hasAcceptedAnswer',
    'numQuestionsAsked30D', 'numHelpProvided30D', 'numQuestionsAsked7D',
    'numHelpProvided7D', 'questionId', 'tag_ids',
]

# Columns added by the revision preprocessing (observable-selection covariates,
# answer quality, ViewCount). Loaded and carried into matched_questions.parquet
# when present; absent columns are tolerated so older processed files still run.
OPTIONAL_COLUMNS = [
    'postHour', 'postDayOfWeek', 'numTags', 'viewCount',
    'bodyLenChars', 'bodyLenWords', 'numCodeBlocks', 'numInlineCode', 'codeLenChars',
    'numParagraphs', 'numLinks', 'numImages', 'numLists', 'numBlockquotes',
    'numSentences', 'numLongWords', 'avgWordLenChars',
    'titleLenChars', 'titleLenWords', 'titleIsQuestion',
    'firstAnswerScore', 'firstAnswerVoteCount',
]

CONTINUOUS_COVS = [
    'timeSinceFirstActivityDays', 'numQuestionsAskedAT', 'numHelpProvidedAT',
    'numQuestionsAsked30D', 'numHelpProvided30D', 'numQuestionsAsked7D',
    'numHelpProvided7D'
]

BUCKET_LABELS = [
    "< 1 Week", "1 Week - 1 Month", "1 - 6 Months", "6 - 12 Months",
    "1 - 3 Years", "3 - 6 Years", "> 6 Years"
]

def create_tenure_buckets(df):
    bins = [-np.inf, 7, 30, 180, 365, 1095, 2190, np.inf]
    df['tenure_bucket'] = pd.cut(
        df['timeSinceFirstActivityDays'], 
        bins=bins, 
        labels=BUCKET_LABELS, 
        right=True
    )
    return df


def _jaccard_tag_overlap(a, b):
    """Jaccard similarity of two tag-ID collections (lists/sets). Handles None/NaN/empty."""
    if a is None or (isinstance(a, float) and np.isnan(a)):
        a = []
    if b is None or (isinstance(b, float) and np.isnan(b)):
        b = []
    if not isinstance(a, (list, tuple)):
        a = [] if not hasattr(a, '__iter__') else list(a)
    if not isinstance(b, (list, tuple)):
        b = [] if not hasattr(b, '__iter__') else list(b)
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def match_single_year_group(year, year_data, treatment_col, z_covariates, caliper,
                            tag_ids_col=None, max_neighbors_for_tiebreak=100):
    """
    Worker function to perform PSM.
    If tag_ids_col is set, among controls within caliper the one with highest
    Jaccard tag overlap to the treated unit is chosen (tie-breaker); controls
    are used at most once when possible.
    Returns a list of dicts mapping global_index to a unique match_id.
    """
    if year_data[treatment_col].nunique() < 2:
        return []

    # Propensity Score Estimation (sklearn to avoid statsmodels/scipy compatibility issues)
    try:
        X = year_data[z_covariates].astype(np.float64)
        y = year_data[treatment_col]
        if y.nunique() < 2:
            return []
        ps_model = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)
        year_data = year_data.copy()
        year_data['propensity_score'] = ps_model.predict_proba(X)[:, 1]
    except Exception:
        return []

    treated = year_data[year_data[treatment_col] == 1].reset_index(drop=True)
    control = year_data[year_data[treatment_col] == 0].reset_index(drop=True)

    if len(control) < 1 or len(treated) < 1:
        return []

    # Matching: get enough neighbors so we can apply caliper + tie-break
    n_neighbors = min(len(control), max_neighbors_for_tiebreak)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(
        control[['propensity_score']]
    )
    distances, indices = nbrs.kneighbors(treated[['propensity_score']])

    match_records = []
    pair_counter = 0
    used_control_positions = set()

    for i in range(len(treated)):
        dist_i = distances[i]
        ind_i = indices[i]
        # Candidates: controls within caliper (by propensity distance)
        if caliper is not None:
            within = np.where(dist_i <= caliper)[0]
        else:
            within = np.arange(len(ind_i))
        if len(within) == 0:
            continue

        # Among candidates, choose by Jaccard tag overlap if available
        if tag_ids_col and tag_ids_col in treated.columns and tag_ids_col in control.columns:
            t_tags = treated.iloc[i][tag_ids_col]
            # (within_idx, control_df_position, jaccard)
            candidates_with_j = [
                (j, ind_i[j], _jaccard_tag_overlap(t_tags, control.iloc[ind_i[j]][tag_ids_col]))
                for j in within
            ]
            # Prefer unused controls (False < True), then higher Jaccard first
            candidates_with_j.sort(key=lambda x: (x[1] in used_control_positions, -x[2]))
            best_j, c_pos_in_control, _ = candidates_with_j[0]
        else:
            # No tie-breaker: use nearest within caliper (first in list)
            best_j = within[0]
            c_pos_in_control = ind_i[best_j]

        used_control_positions.add(c_pos_in_control)
        match_id = f"{year}_{pair_counter}"
        t_indices = treated.iloc[i]['all_indices']
        c_indices = control.iloc[c_pos_in_control]['all_indices']
        for idx in t_indices:
            match_records.append({'global_index': idx, 'match_id': match_id})
        for idx in c_indices:
            match_records.append({'global_index': idx, 'match_id': match_id})
        pair_counter += 1
    return match_records


def match_batch_of_strata(batch_of_groups, exact_match_col, treatment_col, z_covariates, caliper,
                          tag_ids_col=None, max_neighbors_for_tiebreak=100):
    """
    Run match_single_year_group on each stratum in the batch; return flattened list of match records.
    Reduces parallel overhead by processing many strata per worker task.
    """
    out = []
    for group in batch_of_groups:
        stratum_id = group[exact_match_col].iloc[0]
        out.extend(
            match_single_year_group(
                year=stratum_id,
                year_data=group,
                treatment_col=treatment_col,
                z_covariates=z_covariates,
                caliper=caliper,
                tag_ids_col=tag_ids_col,
                max_neighbors_for_tiebreak=max_neighbors_for_tiebreak,
            )
        )
    return out


def perform_psm_matching_phase1_only(data, treatment_col, continuous_covariates,
                                     exact_match_col='exact_match_group', group_col='questionId',
                                     caliper=0.05, n_jobs=-1,
                                     tag_covariates=None, tag_ids_col='tag_ids',
                                     max_neighbors_for_tiebreak=MAX_NEIGHBORS_FOR_TIEBREAK):
    print(f"\n--- Starting PSM at question level on {exact_match_col} ---")
    if tag_covariates:
        print(f"    Propensity model includes {len(tag_covariates)} tag dummies.")
    if tag_ids_col and tag_ids_col in data.columns:
        print(f"    Tie-breaker: Jaccard tag overlap (max_neighbors={max_neighbors_for_tiebreak}).")

    work_df = data.copy().reset_index(drop=True)
    work_df['global_index'] = work_df.index

    dropna_cols = [treatment_col, exact_match_col, group_col] + continuous_covariates
    clean_data = work_df.dropna(subset=dropna_cols).copy()
    n_dropped = len(work_df) - len(clean_data)
    print(f"  Rows with complete covariates: {len(clean_data):,} (dropped {n_dropped:,} missing)")

    # One row per question: index_map gives global_index list (one element per question) per (question, exact_match_group)
    index_map = clean_data.groupby([group_col, exact_match_col])['global_index'].apply(list).reset_index()
    index_map.rename(columns={'global_index': 'all_indices'}, inplace=True)

    unique_data = pd.merge(clean_data, index_map, on=[group_col, exact_match_col], how='inner')
    unique_data = unique_data.drop_duplicates(subset=[group_col, exact_match_col])
    n_questions = len(unique_data)
    n_groups_psm = unique_data[exact_match_col].nunique()
    print(f"  Unique questions (for PS estimation): {n_questions:,}; strata (exact-match groups): {n_groups_psm:,}")

    # Standardization of continuous covariates
    z_covariates = []
    for col in continuous_covariates:
        unique_data[col] = pd.to_numeric(unique_data[col], errors='coerce')
        mu, sigma = unique_data[col].mean(), unique_data[col].std()
        z_col = f"z_{col}"
        z_covariates.append(z_col)
        unique_data[z_col] = (unique_data[col] - mu) / sigma if sigma != 0 else 0.0

    # Propensity formula: z_covariates + tag_covariates (no standardization for binary tag dummies)
    ps_covariates = z_covariates + (tag_covariates if tag_covariates else [])
    unique_data = unique_data.dropna(subset=z_covariates)
    n_after_z = len(unique_data)
    if n_after_z < n_questions:
        print(f"  After dropping NaN in z-covariates: {n_after_z:,} questions ({n_questions - n_after_z:,} dropped)")

    # Parallel execution: batch strata so each task runs many strata (fewer pickles, better load balance)
    grouped_data = [group for _, group in unique_data.groupby(exact_match_col)]
    n_strata = len(grouped_data)
    n_workers = multiprocessing.cpu_count() if n_jobs == -1 else min(n_jobs, multiprocessing.cpu_count())
    n_workers = max(1, n_workers)
    batch_size = max(1, min(PSM_STRATA_BATCH_SIZE, n_strata // max(1, n_workers * 4)))
    batches = [grouped_data[i : i + batch_size] for i in range(0, n_strata, batch_size)]
    tag_col = tag_ids_col if (tag_ids_col and tag_ids_col in unique_data.columns) else None
    print(f"  Fitting propensity and matching within {n_strata:,} strata in {len(batches):,} batches (n_jobs={n_jobs}, batch_size={batch_size})...")
    results = Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="2*n_jobs")(
        delayed(match_batch_of_strata)(
            batch,
            exact_match_col=exact_match_col,
            treatment_col=treatment_col,
            z_covariates=ps_covariates,
            caliper=caliper,
            tag_ids_col=tag_col,
            max_neighbors_for_tiebreak=max_neighbors_for_tiebreak,
        )
        for batch in batches
    )
    
    # Flatten results into a mapping DataFrame (each result is one batch's list of match records)
    match_mapping = pd.DataFrame([item for sublist in results for item in sublist])
    n_batches_with_matches = sum(1 for r in results if len(r) > 0)
    print(f"  Batches with ≥1 match: {n_batches_with_matches:,} / {len(batches):,}")

    if match_mapping.empty:
        print("  Error: No matches found.")
        return pd.DataFrame()

    n_pairs = match_mapping['match_id'].nunique()
    n_matched_rows = len(match_mapping)
    print(f"  Matches: {n_pairs:,} pairs → {n_matched_rows:,} rows (each pair = 1 treated + 1 control question)")

    # Merge the match_id back to the original work_df
    final_matched_df = pd.merge(match_mapping, work_df, on='global_index', how='inner')
    n_t_matched = (final_matched_df[treatment_col] == 1).sum()
    n_c_matched = (final_matched_df[treatment_col] == 0).sum()
    print(f"  Matched sample: treated (hasAnswer=1) {n_t_matched:,}, control (hasAnswer=0) {n_c_matched:,}")

    gc.collect()
    return final_matched_df

def save_balance_latex(df_unmatched, df_matched, treatment_col, covariates, filename="balance_check.tex"):
    def get_stats(df, cov, t_col):
        data = df[df['phase'] == 1] if 'phase' in df.columns else df
        t_data = data[data[t_col] == 1][cov].dropna()
        c_data = data[data[t_col] == 0][cov].dropna()
        if len(t_data) == 0 or len(c_data) == 0: return np.nan, np.nan, np.nan
        t_mean, c_mean = t_data.mean(), c_data.mean()
        pooled_sd = np.sqrt((t_data.var() + c_data.var()) / 2)
        return t_mean, c_mean, (t_mean - c_mean) / pooled_sd if pooled_sd != 0 else 0.0

    with open(filename, "w") as f:
        f.write(r"\begin{table}[htbp]\centering\small" + "\n")
        f.write(r"\begin{tabular}{lcccccc}\toprule" + "\n")
        f.write(r" & \multicolumn{3}{c}{Unmatched} & \multicolumn{3}{c}{Matched} \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}" + "\n")
        f.write(r" Covariate & Tr Mean & Ct Mean & SMD & Tr Mean & Ct Mean & SMD \\\midrule" + "\n")
        for cov in covariates:
            u_t, u_c, u_smd = get_stats(df_unmatched, cov, treatment_col)
            m_t, m_c, m_smd = get_stats(df_matched, cov, treatment_col)
            f.write(f" {cov.replace('_', ' ')} & {u_t:.2f} & {u_c:.2f} & {u_smd:.3f} & {m_t:.2f} & {m_c:.2f} & {m_smd:.3f} \\\\\n")
        f.write(r"\bottomrule\end{tabular}\end{table}" + "\n")

def plot_psm_diagnostics_phase1(original_df, matched_df, treatment_col, continuous_covariates):
    _orig = original_df[original_df['phase'] == 1] if 'phase' in original_df.columns else original_df
    _matched = matched_df[matched_df['phase'] == 1] if 'phase' in matched_df.columns else matched_df
    orig_p1 = _orig.dropna(subset=[treatment_col] + continuous_covariates).copy()
    matched_p1 = _matched.copy()

    X_orig = orig_p1[continuous_covariates].astype(np.float64)
    model = LogisticRegression(max_iter=1000, random_state=0).fit(X_orig, orig_p1[treatment_col])
    orig_p1['ps'] = model.predict_proba(X_orig)[:, 1]
    valid = matched_p1[continuous_covariates].notna().all(axis=1)
    matched_p1['ps'] = np.nan
    if valid.any():
        matched_p1.loc[valid, 'ps'] = model.predict_proba(
            matched_p1.loc[valid, continuous_covariates].astype(np.float64)
        )[:, 1]

    plt.figure(figsize=(10, 5))
    sns.kdeplot(orig_p1[orig_p1[treatment_col]==0]['ps'], label='Control (Orig)', color='grey', fill=True)
    sns.kdeplot(orig_p1[orig_p1[treatment_col]==1]['ps'], label='Treated (Orig)', color='blue', fill=True)
    sns.kdeplot(matched_p1[matched_p1[treatment_col]==0]['ps'], label='Control (Matched)', color='red', linestyle='--')
    plt.title('Propensity Score Support')
    plt.legend(); plt.savefig(PLOT_COMMON_SUPPORT); plt.close()

def generate_love_plot(df_unmatched, df_matched, treatment_col, covariates, save_path):
    """
    Generates a Love Plot (Covariate Balance Plot) comparing ASMD before and after matching.
    """
    print(f"Generating Love Plot at {save_path}...")
    
    def calculate_smd(df, cov):
        data = df[df['phase'] == 1] if 'phase' in df.columns else df
        t = data[data[treatment_col] == 1][cov]
        c = data[data[treatment_col] == 0][cov]
        
        if len(t) < 2 or len(c) < 2:
            return 0.0
            
        diff = t.mean() - c.mean()
        pooled_var = (t.var() + c.var()) / 2
        return abs(diff / np.sqrt(pooled_var)) if pooled_var > 0 else 0.0

    records = []
    for cov in covariates:
        u_smd = calculate_smd(df_unmatched, cov)
        m_smd = calculate_smd(df_matched, cov)
        
        # Add readable labels
        clean_name = cov.replace("num", "# ").replace("QuestionsAsked", "Q Asked").replace("HelpProvided", "Help Provided")
        
        records.append({'Covariate': clean_name, 'Abs_SMD': u_smd, 'Dataset': 'Unmatched'})
        records.append({'Covariate': clean_name, 'Abs_SMD': m_smd, 'Dataset': 'Matched'})

    plot_df = pd.DataFrame(records)
    
    # Sort by Unmatched SMD for better visual hierarchy
    sort_order = plot_df[plot_df['Dataset'] == 'Unmatched'].sort_values('Abs_SMD', ascending=True)['Covariate'].tolist()

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Create the scatter plot
    ax = sns.scatterplot(
        data=plot_df, 
        y='Covariate', 
        x='Abs_SMD', 
        hue='Dataset', 
        style='Dataset',
        s=100, 
        palette={'Unmatched': 'grey', 'Matched': 'red'},
        markers={'Unmatched': 'o', 'Matched': 'X'}
    )
    
    # Add threshold line
    plt.axvline(x=0.1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.text(0.105, 0.5, 'Threshold (0.1)', rotation=90, verticalalignment='center', alpha=0.7)
    
    # Set labels and title
    plt.title('Covariate Balance (Love Plot)', fontsize=14)
    plt.xlabel('Absolute Standardized Mean Difference (ASMD)')
    plt.ylabel('')
    
    # Reorder Y-axis based on sort_order
    plt.yticks(range(len(sort_order)), sort_order)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Love Plot saved.")

def _load_tag_accept_shares_cache(cache_path, data_path, per_year, min_count):
    """
    Load cached tag accept shares if cache exists and matches (same data file mtime and params).
    Returns (shares_dict, per_year_used) or (None, None) on miss.
    """
    if not cache_path or not os.path.isfile(cache_path):
        return None, None
    try:
        data_mtime = os.path.getmtime(data_path)
    except OSError:
        return None, None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None, None
    if not isinstance(payload, dict) or "params" not in payload or "shares" not in payload:
        return None, None
    cached_path, cached_mtime, cached_per_year, cached_min = payload["params"]
    if (str(cached_path) != str(data_path) or cached_mtime != data_mtime or
            cached_per_year != per_year or cached_min != min_count):
        return None, None
    return payload["shares"], cached_per_year


def _save_tag_accept_shares_cache(cache_path, data_path, per_year, min_count, shares_dict):
    """Save tag accept shares to cache with current data path, mtime, and params."""
    try:
        data_mtime = os.path.getmtime(data_path)
    except OSError:
        return
    payload = {
        "params": (str(data_path), data_mtime, per_year, min_count),
        "shares": shares_dict,
    }
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)
    except Exception:
        pass


def _safe_tag_list(x):
    """Return tag_ids as a set for membership tests; handle None/NaN/empty.
    Handles list/tuple, array-like (e.g. numpy/pyarrow from parquet), and string-serialized lists."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()
    if isinstance(x, str) and x.strip().startswith('['):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple)):
                return set(int(t) for t in parsed)
        except (ValueError, SyntaxError, TypeError):
            pass
        return set()
    try:
        if isinstance(x, (list, tuple)) or hasattr(x, '__iter__'):
            return set(int(t) for t in x)
    except (TypeError, ValueError):
        pass
    return set()


def compute_tag_accept_shares(df, tag_ids_col='tag_ids', outcome_col='hasAcceptedAnswer', year_col='year',
                              min_count=100):
    """
    For each tag, compute the share of questions with that tag that have outcome_col==1.

    Only (year, tag_id) or tag_id with at least min_count occurrences are included
    (per year when year_col is set, else overall). Others are omitted so lookup yields NaN and caller can impute.

    If year_col is None: use all questions; return dict tag_id -> share (tags with count >= min_count).
    If year_col is set: restrict to same calendar year per tag; return dict (year, tag_id) -> share.
    """
    need_year = year_col is not None and year_col in df.columns
    cols = [tag_ids_col, outcome_col]
    if need_year:
        cols.append(year_col)
    subset = df[cols].dropna(subset=[tag_ids_col, outcome_col])
    # Validate tag parsing on first row
    if len(subset) > 0:
        sample = subset.iloc[0]
        raw = sample[tag_ids_col]
        parsed = _safe_tag_list(raw)
        print(f"    [validate] sample row: tag_ids type={type(raw).__name__}, raw (repr)={repr(raw)[:80]}..., parsed set size={len(parsed)}, sample ids={list(parsed)[:5]}")
    rows = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Tag accept shares", unit="rows"):
        tags = _safe_tag_list(row[tag_ids_col])
        y = int(row[outcome_col])
        yr = row[year_col] if need_year else None
        for t in tags:
            if need_year:
                rows.append({'year': yr, 'tag_id': t, 'y': y})
            else:
                rows.append({'tag_id': t, 'y': y})
    if not rows:
        return {}
    expand = pd.DataFrame(rows)
    n_expand_rows = len(expand)
    if need_year:
        agg = expand.groupby(['year', 'tag_id'])['y'].agg(['mean', 'count'])
        n_before = len(agg)
        # Only keep (year, tag_id) with at least min_count occurrences within that year
        agg = agg[agg['count'] >= min_count]
        n_after = len(agg)
        print(f"    [validate] expand: {n_expand_rows:,} rows; (year,tag_id) groups: {n_before:,} before count>={min_count}, {n_after:,} after.")
        result = {(int(yr), int(t)): agg.loc[(yr, t), 'mean'] for (yr, t) in agg.index}
    else:
        agg = expand.groupby('tag_id')['y'].agg(['mean', 'count'])
        n_before = len(agg)
        agg = agg[agg['count'] >= min_count]
        n_after = len(agg)
        print(f"    [validate] expand: {n_expand_rows:,} rows; tag_id groups: {n_before:,} before count>={min_count}, {n_after:,} after.")
        result = agg['mean'].to_dict()
    # Print first few tag (or (year,tag)) entries to validate
    n_show = 5
    if need_year and result:
        keys_sorted = sorted(result.keys())[:n_show]
        print(f"    [validate] first {n_show} (year, tag_id) accept shares:")
        for k in keys_sorted:
            yr, tid = k
            share = result[k]
            cnt = agg.loc[k, 'count'] if k in agg.index else None
            print(f"      (year={yr}, tag_id={tid}): share={share:.4f}, count={cnt}")
    elif result:
        keys_sorted = sorted(result.keys(), key=lambda t: -agg.loc[t, 'count'])[:n_show]  # show by count desc
        print(f"    [validate] first {n_show} tag_id accept shares (by count):")
        for tid in keys_sorted:
            share = result[tid]
            cnt = agg.loc[tid, 'count']
            print(f"      tag_id={tid}: share={share:.4f}, count={int(cnt):,}")
    return result


def add_tag_accept_share_covariates(df, tag_accept_share_by_id, tag_ids_col='tag_ids',
                                    year_col=None, per_year=False, avg_only=True):
    """
    Add continuous tag covariates: for each question, average (and optionally max) of
    accept-share over its tags. Questions with no tags get np.nan (caller should impute if needed).

    If per_year is True, year_col must be set and tag_accept_share_by_id is keyed by (year, tag_id).
    """
    def avg_share(tag_list, year=None):
        tags = _safe_tag_list(tag_list)
        if not tags:
            return np.nan
        if per_year and year is not None:
            vals = [tag_accept_share_by_id.get((int(year), t), np.nan) for t in tags]
        else:
            vals = [tag_accept_share_by_id.get(t, np.nan) for t in tags]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return np.mean(vals) if vals else np.nan

    def max_share(tag_list, year=None):
        tags = _safe_tag_list(tag_list)
        if not tags:
            return np.nan
        if per_year and year is not None:
            vals = [tag_accept_share_by_id.get((int(year), t), np.nan) for t in tags]
        else:
            vals = [tag_accept_share_by_id.get(t, np.nan) for t in tags]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return np.max(vals) if vals else np.nan

    df = df.copy()
    if per_year and year_col and year_col in df.columns:
        tqdm.pandas(desc="Tag share (avg)", unit="rows")
        df['tag_accept_share_avg'] = df.progress_apply(
            lambda r: avg_share(r[tag_ids_col], r.get(year_col)), axis=1
        )
        if not avg_only:
            tqdm.pandas(desc="Tag share (max)", unit="rows")
            df['tag_accept_share_max'] = df.progress_apply(
                lambda r: max_share(r[tag_ids_col], r.get(year_col)), axis=1
            )
    else:
        df['tag_accept_share_avg'] = df[tag_ids_col].map(lambda x: avg_share(x, None))
        if not avg_only:
            df['tag_accept_share_max'] = df[tag_ids_col].map(lambda x: max_share(x, None))
    return df


def main():
    data_path = str(DATA_PATH)
    print(f"\n{'='*60}")
    print("MATCHING PIPELINE: Question-centered PSM (hasAnswer)")
    print(f"{'='*60}\n")
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: file not found: {data_path}")
        sys.exit(1)
    # Request columns that exist (parquet may use mainTagId or main_tag_id)
    import pyarrow.parquet as pq
    parquet_names = set(pq.read_schema(data_path).names)
    available_optional = [c for c in OPTIONAL_COLUMNS if c in parquet_names]
    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in parquet_names]
    if missing_optional:
        print(f"  Note: optional covariate columns absent from parquet (older preprocessing run?): {missing_optional}")
    df = pd.read_parquet(data_path, columns=REQUIRED_COLUMNS + available_optional)
    def _first_tag_id(x):
        s = _safe_tag_list(x)
        return next(iter(s), pd.NA) if s else pd.NA
    df['mainTagId'] = df['tag_ids'].apply(_first_tag_id)
    if df['mainTagId'].notna().any():
        print(f"  Derived mainTagId from tag_ids (first tag per row).")
    print(df['mainTagId'].head())
    print(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")
    n_before_self = len(df)
    df = df[df['hasSelfAnswer'] == 0].reset_index(drop=True)
    print(f"  After dropping hasSelfAnswer==1: {len(df):,} rows ({n_before_self - len(df):,} removed)")
    df['responseTimeDays'] = df['responseTimeHours'] / 24
    n_recode = (df['responseTimeDays'] > 7) & (df['hasAnswer'] == 1)
    df.loc[df['responseTimeDays'] > 7, 'hasAnswer'] = 0
    print(f"  Recoded hasAnswer=0 for responseTime>7d: {n_recode.sum():,} rows")
    df = create_tenure_buckets(df)
    n_treated = (df['hasAnswer'] == 1).sum()
    n_control = (df['hasAnswer'] == 0).sum()
    print(f"  Treatment: hasAnswer=1 → {n_treated:,}, hasAnswer=0 → {n_control:,} (total {len(df):,})")

    # Observable-selection covariates (R1 Q3): posting time enters cyclically /
    # binned, question length as log. ViewCount is deliberately NOT a propensity
    # covariate (it accrues after treatment); it is only carried through for the
    # placebo analysis. Only covariates present in the parquet are used.
    extra_matching_covs = []
    if 'postHour' in df.columns:
        hours = pd.to_numeric(df['postHour'], errors='coerce')
        df['postHourSin'] = np.sin(2 * np.pi * hours / 24)
        df['postHourCos'] = np.cos(2 * np.pi * hours / 24)
        extra_matching_covs += ['postHourSin', 'postHourCos']
    if 'postDayOfWeek' in df.columns:
        df['isWeekend'] = (pd.to_numeric(df['postDayOfWeek'], errors='coerce') >= 5).astype(float)
        extra_matching_covs.append('isWeekend')
    if 'numTags' in df.columns:
        df['numTags'] = pd.to_numeric(df['numTags'], errors='coerce')
        extra_matching_covs.append('numTags')
    if 'bodyLenChars' in df.columns:
        df['logBodyLenChars'] = np.log1p(pd.to_numeric(df['bodyLenChars'], errors='coerce'))
        extra_matching_covs.append('logBodyLenChars')
    if extra_matching_covs:
        print(f"  Additional propensity covariates (observable selection): {extra_matching_covs}")
    else:
        print("  No additional observable-selection covariates available in input parquet.")

    # Tag-related covariates for propensity score
    print(f"\n--- Tag covariates ---")
    tag_dict_path = str(TAG_DICTIONARY_PATH)
    if not os.path.exists(tag_dict_path):
        raise FileNotFoundError(
            f"Tag dictionary not found: {tag_dict_path}. "
            "Run creating_raw_reciprocity_dataset.py first."
        )
    tag_dict = pd.read_parquet(tag_dict_path)
    top_tag_ids = (
        tag_dict.sort_values('tag_frequency', ascending=False)
        .head(TOP_K_TAGS)['tag_id']
        .tolist()
    )
    print(f"  Tag dictionary: {len(tag_dict):,} tags; using top-{len(top_tag_ids)} for tie-break.")

    if USE_TAG_ACCEPT_SHARE:
        # Replace 50 one-hot tag dummies with 1–2 continuous vars: share of questions with accepted answer per tag.
        cache_path = str(TAG_ACCEPT_SHARES_CACHE_PATH)
        tag_accept_share_by_id, tag_accept_share_per_year = _load_tag_accept_shares_cache(
            cache_path, data_path, TAG_ACCEPT_SHARE_PER_YEAR, MIN_TAG_COUNT_FOR_SHARE
        )
        from_cache = tag_accept_share_by_id is not None
        if from_cache:
            n_share_entries = len(tag_accept_share_by_id)
            shares = list(tag_accept_share_by_id.values())
            print(f"  Tag accept shares: loaded from cache ({n_share_entries:,} entries, per_year={tag_accept_share_per_year}); "
                  f"range [{min(shares):.3f}, {max(shares):.3f}], mean={np.mean(shares):.3f}")
        else:
            tag_accept_share_per_year = TAG_ACCEPT_SHARE_PER_YEAR
            print(f"  Computing tag accept shares (outcome=hasAcceptedAnswer, per_year={tag_accept_share_per_year}, "
                  f"min_count={MIN_TAG_COUNT_FOR_SHARE} per year/tag)...")
            tag_accept_share_by_id = compute_tag_accept_shares(
                df, tag_ids_col='tag_ids', outcome_col='hasAcceptedAnswer',
                year_col='year' if tag_accept_share_per_year else None,
                min_count=MIN_TAG_COUNT_FOR_SHARE
            )
            if not tag_accept_share_by_id and tag_accept_share_per_year:
                print(f"  No (year,tag) cells with count>={MIN_TAG_COUNT_FOR_SHARE}; falling back to overall tag shares (no year split).")
                tag_accept_share_by_id = compute_tag_accept_shares(
                    df, tag_ids_col='tag_ids', outcome_col='hasAcceptedAnswer',
                    year_col=None,
                    min_count=MIN_TAG_COUNT_FOR_SHARE
                )
                tag_accept_share_per_year = False
            if tag_accept_share_by_id:
                _save_tag_accept_shares_cache(
                    cache_path, data_path, tag_accept_share_per_year, MIN_TAG_COUNT_FOR_SHARE, tag_accept_share_by_id
                )
                print(f"  Cached tag accept shares to {cache_path}")
        n_share_entries = len(tag_accept_share_by_id) if tag_accept_share_by_id else 0
        if tag_accept_share_by_id and not from_cache:
            shares = list(tag_accept_share_by_id.values())
            print(f"  Tag accept shares: {n_share_entries:,} (year,tag) or tag entries (count>={MIN_TAG_COUNT_FOR_SHARE}); "
                  f"share range [{min(shares):.3f}, {max(shares):.3f}], mean={np.mean(shares):.3f}")
        elif not tag_accept_share_by_id:
            print(f"  Warning: no tag accept shares computed (empty dict).")
        df = add_tag_accept_share_covariates(
            df, tag_accept_share_by_id, tag_ids_col='tag_ids',
            year_col='year' if tag_accept_share_per_year else None,
            per_year=tag_accept_share_per_year,
            avg_only=TAG_ACCEPT_SHARE_AVG_ONLY
        )
        # Validate: print first few rows with tag_ids, hasAcceptedAnswer, tag_accept_share_avg
        if 'tag_accept_share_avg' in df.columns:
            cols_show = ['tag_ids', 'hasAcceptedAnswer', 'tag_accept_share_avg']
            if 'year' in df.columns:
                cols_show = ['year'] + cols_show
            available = [c for c in cols_show if c in df.columns]
            sample = df[available].head(5)
            print(f"  [validate] sample rows (first 5) after adding tag_accept_share:")
            for _, row in sample.iterrows():
                tag_ids_str = str(row.get('tag_ids', ''))[:60]
                yr = row.get('year', 'n/a')
                acc = row.get('hasAcceptedAnswer')
                avg = row.get('tag_accept_share_avg')
                print(f"    year={yr} tag_ids={tag_ids_str}... hasAcceptedAnswer={acc} tag_accept_share_avg={avg}")
        # Impute missing (no tags or no data for that year/tag) with overall accept rate
        for col in ['tag_accept_share_avg', 'tag_accept_share_max']:
            if col in df.columns and df[col].isna().any():
                n_missing = df[col].isna().sum()
                fill_val = df['hasAcceptedAnswer'].mean()
                df[col] = df[col].fillna(fill_val)
                print(f"  Imputed {n_missing:,} missing {col} with overall hasAcceptedAnswer mean = {fill_val:.3f}")
        if 'tag_accept_share_avg' in df.columns:
            print(f"  tag_accept_share_avg: mean={df['tag_accept_share_avg'].mean():.3f}, "
                  f"std={df['tag_accept_share_avg'].std():.3f}, "
                  f"min={df['tag_accept_share_avg'].min():.3f}, max={df['tag_accept_share_avg'].max():.3f}")
        tag_share_cols = ['tag_accept_share_avg']
        if not TAG_ACCEPT_SHARE_AVG_ONLY and 'tag_accept_share_max' in df.columns:
            tag_share_cols.append('tag_accept_share_max')
        continuous_covariates = CONTINUOUS_COVS + tag_share_cols
        tag_covariates = []
        time_scope = "within same calendar year" if tag_accept_share_per_year else "all years"
        print(f"  PSM tag covariates: {tag_share_cols} (share with accepted answer per tag, {time_scope})")
    else:
        # Legacy: top-K one-hot tag dummies
        for i, tid in tqdm(enumerate(top_tag_ids), total=len(top_tag_ids), desc="Tag dummies", unit="tag"):
            df[f'tag_{i}'] = df['tag_ids'].apply(
                lambda x, t=tid: 1 if t in _safe_tag_list(x) else 0
            )
        tag_covariates = [f'tag_{i}' for i in range(len(top_tag_ids))]
        continuous_covariates = CONTINUOUS_COVS
        print(f"PSM tag covariates: {len(tag_covariates)} tag dummies (legacy)")

    # Observable-selection covariates (R1 Q3) enter the propensity model alongside
    # the activity-history covariates
    continuous_covariates = continuous_covariates + extra_matching_covs

    # Exact-match: same year AND same main tag (treated and controls only matched within stratum)
    main_tag_str = df['mainTagId'].astype("Int64").astype(str).replace("<NA>", "NA").replace("nan", "NA").fillna("NA")
    df['exact_match_group'] = df['year'].astype(str) + "_" + main_tag_str
    n_groups = df['exact_match_group'].nunique()
    years = sorted(df['year'].dropna().unique().tolist())
    n_main_tags = df['mainTagId'].notna().sum()
    print(f"\n--- Exact-match groups (year + mainTagId) ---")
    print(f"  Match exactly on: year and mainTagId (no cross-year or cross-tag pairs)")
    print(f"  Unique strata: {n_groups:,}; years: {min(years) if years else 'N/A'}–{max(years) if years else 'N/A'}; questions with mainTagId: {n_main_tags:,}")
    if n_main_tags == 0 and len(df) > 0:
        print(f"  Warning: no questions have non-null mainTagId — matching is effectively by year only. Re-run preprocessing/processing_reciprocity_dataset.py to populate mainTagId from tag_ids.")

    # Exclude year-tag strata with fewer than MIN_QUESTIONS_PER_STRATUM questions from matching
    stratum_counts = df.groupby('exact_match_group').size()
    strata_kept = stratum_counts[stratum_counts >= MIN_QUESTIONS_PER_STRATUM].index
    strata_dropped = stratum_counts[stratum_counts < MIN_QUESTIONS_PER_STRATUM]
    n_strata_dropped = len(strata_dropped)
    n_questions_dropped = strata_dropped.sum()
    df = df[df['exact_match_group'].isin(strata_kept)].reset_index(drop=True)
    if n_strata_dropped > 0:
        print(f"  Excluded {n_strata_dropped:,} year-tag strata with < {MIN_QUESTIONS_PER_STRATUM} questions ({n_questions_dropped:,} rows removed). Remaining: {len(strata_kept):,} strata, {len(df):,} rows.")
    else:
        print(f"  All year-tag strata have ≥ {MIN_QUESTIONS_PER_STRATUM} questions; none excluded.")

    print(f"\n--- PSM matching ---")
    matched_df = perform_psm_matching_phase1_only(
        df, treatment_col='hasAnswer', continuous_covariates=continuous_covariates,
        exact_match_col='exact_match_group', caliper=0.05,
        tag_covariates=tag_covariates, tag_ids_col='tag_ids',
        max_neighbors_for_tiebreak=MAX_NEIGHBORS_FOR_TIEBREAK,
    )

    if not matched_df.empty:
        print(f"\n--- Writing output ---")
        OUTPUT_MATCHED_PATH.parent.mkdir(parents=True, exist_ok=True)
        matched_df.to_parquet(str(OUTPUT_MATCHED_PATH))
        print(f"  Matched data: {OUTPUT_MATCHED_PATH} ({len(matched_df):,} rows, {matched_df['match_id'].nunique():,} pairs)")
        save_balance_latex(df, matched_df, 'hasAnswer', continuous_covariates, filename=str(BALANCE_TEX_PATH))
        print(f"  Balance table (SMD): {BALANCE_TEX_PATH}")
        plot_psm_diagnostics_phase1(df, matched_df, 'hasAnswer', continuous_covariates)
        print(f"  Common-support plot: {PLOT_COMMON_SUPPORT}")
        generate_love_plot(df, matched_df, 'hasAnswer', continuous_covariates, str(PLOT_LOVE))
        print(f"\nDone. Pair examples:\n{matched_df[['match_id', 'hasAnswer', 'questionId']].drop_duplicates('match_id').head()}")
    else:
        print("\nNo matched pairs produced; skipping output.")

if __name__ == "__main__":
    main()