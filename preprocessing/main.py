import os
from pathlib import Path
# import creating_raw_bounty_dataset
import creating_raw_reciprocity_dataset
# import processing_bounty_dataset
import processing_reciprocity_dataset


def main():
    # Setup directories
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir_parent = base_dir.parent

    # Define common data directories
    input_data_folder = base_dir_parent / "data" / "input"
    study_datasets_folder = base_dir_parent / "data" / "study_datasets"
    os.makedirs(input_data_folder, exist_ok=True)
    os.makedirs(study_datasets_folder, exist_ok=True)

    # Configuration parameters
    config = {
        # "bounty_timeline_path": str(input_data_folder / "bounty_timeline.parquet"),
        "window_days": [7],
        "chunk_size": 100000,
        "cutoff_date": "2025-04-01"
    }

    # Step 1: Create raw bounty dataset
    # creating_raw_bounty_dataset.create_user_answers_dataset(
    #     input_folder=str(input_data_folder),
    #     output_folder=str(input_data_folder),
    #     bounty_timeline_path=config["bounty_timeline_path"]
    # )

    # Step 2: Create reciprocity dataset
    for days in config["window_days"]:
        creating_raw_reciprocity_dataset.process_question_data(
            input_folder=str(input_data_folder),
            output_folder=str(input_data_folder),
            window_length=days,
            include_all_questions=True
        )

    # Step 3: Process bounty dataset
    # processing_bounty_dataset.process_bounty_dataset(
    #     input_file=str(input_data_folder / "user_answers_bounty_dataset.parquet"),
    #     output_file=str(study_datasets_folder / "user_answers_bounty_processed.parquet"),
    #     chunk_size=config["chunk_size"]
    # )

    # Step 4: Process reciprocity dataset
    for days in config["window_days"]:
        input_file = str(input_data_folder / f"question_centered_model_{days}d_all_questions.parquet")
        output_file = str(study_datasets_folder / f"question_centered_model_{days}d_processed.parquet")

        processing_reciprocity_dataset.process_question_dataset(
            input_file=input_file,
            output_file=output_file,
            chunk_size=config["chunk_size"],
            cutoff_date=config["cutoff_date"]
        )


if __name__ == "__main__":
    main()