import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker


def set_size(width=433.62, fraction=1, equal_height_to_fraction_1=False):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters:
    width: float
        Document textwidth or columnwidth in pts
    fraction: float
        Fraction of the width which you wish the figure to occupy
    equal_height_to_fraction_1: bool
        Whether to make the height the same as it would be for fraction=1
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    if equal_height_to_fraction_1:
        # Use full width to calculate height if equal_height_to_fraction_1 is True
        fig_height_in = (width * inches_per_pt) * golden_ratio
    else:
        # Use current fraction to calculate height
        fig_height_in = fig_width_in * golden_ratio

    return (fig_width_in, fig_height_in)


def empirical_setting_number_qna():
    """
    Generate Q&A statistics plot from parquet files and save as EPS.
    """

    # Set up LaTeX rendering and styling parameters
    plt.rcParams['text.latex.preamble'] = r"\usepackage{txfonts}"
    params = {
        'text.usetex': True,
        'font.size': 11,
        'font.family': 'serif',
    }
    plt.rcParams.update(params)

    def extract_yearmonth_stats(input_folder):
        """
        Extract monthly statistics from raw parquet files using DuckDB

        Parameters
        ----------
        input_folder: str
            Path to folder containing posts_questions.parquet and posts_answers.parquet

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Year, Month, QuestionsCount, AnswersCount, AcceptedPercentage
        """

        # Initialize DuckDB connection with performance settings
        con = duckdb.connect(database=':memory:')
        con.execute("PRAGMA memory_limit='10GB';")
        con.execute("PRAGMA max_temp_directory_size='200GiB'")
        con.execute("PRAGMA threads=4;")
        con.execute("PRAGMA enable_progress_bar;")

        # Set file paths
        questions_path = os.path.join(input_folder, 'posts_questions.parquet')
        answers_path = os.path.join(input_folder, 'posts_answers.parquet')

        # Query to get monthly statistics in one go
        query = f"""
        WITH monthly_questions AS (
            SELECT 
                YEAR(CAST(CreationDate AS TIMESTAMP)) AS Year,
                MONTH(CAST(CreationDate AS TIMESTAMP)) AS Month,
                COUNT(*) AS QuestionsCount,
                COUNT(AcceptedAnswerId) AS AcceptedCount
            FROM '{questions_path}'
            GROUP BY YEAR(CAST(CreationDate AS TIMESTAMP)), MONTH(CAST(CreationDate AS TIMESTAMP))
        ),
        monthly_answers AS (
            SELECT 
                YEAR(CAST(CreationDate AS TIMESTAMP)) AS Year,
                MONTH(CAST(CreationDate AS TIMESTAMP)) AS Month,
                COUNT(*) AS AnswersCount
            FROM '{answers_path}'
            GROUP BY YEAR(CAST(CreationDate AS TIMESTAMP)), MONTH(CAST(CreationDate AS TIMESTAMP))
        )
        SELECT 
            COALESCE(q.Year, a.Year) AS Year,
            COALESCE(q.Month, a.Month) AS Month,
            COALESCE(q.QuestionsCount, 0) AS QuestionsCount,
            COALESCE(a.AnswersCount, 0) AS AnswersCount,
            COALESCE(q.AcceptedCount, 0) AS AcceptedCount,
            CASE 
                WHEN COALESCE(q.QuestionsCount, 0) > 0 
                THEN (COALESCE(q.AcceptedCount, 0) * 100.0 / q.QuestionsCount)
                ELSE 0 
            END AS AcceptedPercentage
        FROM monthly_questions q
        FULL OUTER JOIN monthly_answers a 
            ON q.Year = a.Year AND q.Month = a.Month
        ORDER BY Year, Month;
        """

        # Execute query and get results
        result_df = con.execute(query).fetchdf()

        # Close connection
        con.close()

        return result_df[['Year', 'Month', 'QuestionsCount', 'AnswersCount', 'AcceptedPercentage']]

    # Set input folder path
    input_folder = r"..\data\input"

    # Extract statistics from parquet files using DuckDB
    print("Extracting statistics using DuckDB queries...")
    merged_df = extract_yearmonth_stats(input_folder)
    print(f"Extracted data for {len(merged_df)} months from {merged_df['Year'].min()} to {merged_df['Year'].max()}")

    # Convert counts to millions
    merged_df['QuestionsCount'] = merged_df['QuestionsCount'] / 1e6
    merged_df['AnswersCount'] = merged_df['AnswersCount'] / 1e6

    # Create a new column for plotting on the x-axis as Year-Month
    merged_df['YearMonth'] = merged_df['Year'].astype(str) + "-" + merged_df['Month'].astype(str)

    # Create a figure and a set of subplots with specific size
    fig, ax1 = plt.subplots(figsize=set_size())

    # Plot Questions and Answers Count as line plots with thinner lines
    ax1.plot(merged_df['YearMonth'], merged_df['QuestionsCount'], label='Number of questions', color='dodgerblue',
             linewidth=2)
    ax1.plot(merged_df['YearMonth'], merged_df['AnswersCount'], label='Number of answers', color='gray', linewidth=2)

    # Set the x-axis and y-axis labels
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Count (in Millions)')
    ax1.grid(False)

    # Filter YearMonth labels to only display the beginning of each year
    first_of_each_year = merged_df[merged_df['Month'] == 1]['YearMonth']
    years = merged_df[merged_df['Month'] == 1]['Year']

    # Create labels that are empty except for every second year
    labels = [year if i % 2 == 0 else '' for i, year in enumerate(years)]

    ax1.set_xticks(first_of_each_year)
    ax1.set_xticklabels(labels, ha='center')

    # Set the x-axis limits to the range of the data
    ax1.set_xlim(merged_df['YearMonth'].iloc[0], merged_df['YearMonth'].iloc[-1])

    # Create a secondary y-axis for the Accepted Answer Rate with thinner lines
    ax2 = ax1.twinx()
    ax2.plot(merged_df['YearMonth'], merged_df['AcceptedPercentage'], color='red', markersize=4,
             label='Accepted answers',
             linestyle=':', linewidth=2)
    ax2.set_ylabel('Rate (in \%)')
    ax2.set_ylim(0, 100)  # Set y-axis limits for the secondary axis
    ax1.set_ylim(0, 0.4)  # Set y-axis limits for the primary axis

    # Set the left y-axis ticks
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))  # Minor ticks at 0.05 intervals

    # Set the right y-axis ticks to 0, 25, 50, 75, 100
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(12.5))  # Minor ticks at 12.5 intervals

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)

    # Save the plot with LaTeX formatting
    plt.savefig("empirical_setting_qna_count.eps", dpi=1000, bbox_inches='tight')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    empirical_setting_number_qna()