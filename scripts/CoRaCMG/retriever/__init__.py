from fire import Fire

from commit import load_from_jsonl
from commit.retriever.embedding import retrieve


def main(
    query_jsonl: str,
    database_jsonl: str,
    output_jsonl: str,
):
    query_instances = load_from_jsonl(query_jsonl)[:2]
    print(f"Loaded {len(query_instances)} query instances.")

    database_instances = load_from_jsonl(database_jsonl)
    print(f"Loaded {len(database_instances)} database instances.")

    retrieved_instances = retrieve(query_instances, database_instances)

    print(f"QUERY DIFF\n{query_instances[1].diff}")
    print(f"QUERY DIFF\n{retrieved_instances[1][0].diff}")


if __name__ == "__main__":
    Fire(main)
