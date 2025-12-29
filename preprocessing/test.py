import duckdb
from pathlib import Path


def test_autobiography_badges():
    """Quick test to check Autobiography badges in the dataset."""

    con = duckdb.connect(database=':memory:')

    base_dir = Path("..")
    badges_path = base_dir / "data" / "input" / "Badges.parquet"

    print(f"Loading badges from: {badges_path}")

    # Load Autobiography badges
    con.execute(f"""
        CREATE TEMPORARY VIEW autobiography_badges AS
        SELECT
            UserId AS user_id,
            CAST(Date AS TIMESTAMP) AS autobiography_received
        FROM '{badges_path}'
        WHERE Name = 'Autobiography';
    """)

    # Count Autobiography badges
    result = con.execute("SELECT COUNT(*) as count FROM autobiography_badges").fetchone()
    autobiography_count = result[0]

    print(f"\nAutobiography badges found: {autobiography_count:,}")

    # If no Autobiography badges found, show all unique badge names
    if autobiography_count == 0:
        print("\nNo Autobiography badges found. Listing all unique badge names:")
        badge_names = con.execute(f"""
            SELECT DISTINCT Name 
            FROM '{badges_path}' 
            ORDER BY Name
        """).fetchdf()

        for idx, name in enumerate(badge_names['Name'], 1):
            print(f"  {idx}. {name}")
    else:
        # Show sample of Autobiography badges
        print("\nSample Autobiography badges:")
        sample = con.execute("""
                             SELECT *
                             FROM autobiography_badges LIMIT 5
                             """).fetchdf()
        print(sample)

    con.close()


if __name__ == "__main__":
    test_autobiography_badges()