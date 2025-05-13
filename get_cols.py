import pyarrow.parquet as pq

# Get schema for the main processed file
schema1 = pq.read_schema("data/study_datasets/user_answers_bounty_processed.parquet")
print("Schema for user_answers_bounty_processed.parquet:")
print(schema1.names)

print("\n" + "="*50 + "\n")

# Get schema for the chunk file
# Based on your processing script, chunks are saved in {output_file}_chunks directory
chunk_schema = pq.read_schema("data/study_datasets/question_centered_model_7d_processed.parquet_chunks/chunk_00.parquet")
print("Schema for chunk_00.parquet:")
print(chunk_schema.names)