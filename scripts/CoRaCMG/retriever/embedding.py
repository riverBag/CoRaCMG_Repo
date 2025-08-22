import numpy as np
import torch
from fire import Fire
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from commit import Instance, load_from_jsonl

BATCH_SIZE = 32


def get_embeddings(diffs: list[str]) -> list[list[float]]:
    """
    Get embeddings for a list of diffs in a batch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-code", trust_remote_code=True
    )
    model.max_seq_length = 8192

    embeddings = model.encode(
        diffs,
        show_progress_bar=True,
        device=device,
        convert_to_tensor=True,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
    )

    embeddings = [emb.tolist() for emb in embeddings]

    return embeddings


def retrieve(
    instances: list[Instance], database: list[Instance], k: int = 1
) -> list[list[Instance]]:
    """
    Retrieve the top k most similar instances from the database for each instance in the input list.
    Based on the pre-calculated semantic embeddings.
    """
    instance_embeddings = np.array([inst.embedding for inst in instances])
    database_embeddings = np.array([db_inst.embedding for db_inst in database])

    print(
        f"Calculating similarity matrix with shape ({len(instances)} x {len(database)})... This may take a while."
    )
    similarity_matrix = np.dot(instance_embeddings, database_embeddings.T)

    results = []
    for i in tqdm(range(len(instances)), desc="Retrieving"):
        current_id = instances[i].instance_id
        scores = similarity_matrix[i]

        candidate_indices = np.argpartition(scores, -2 * k)[-2 * k :]
        sorted_candidates = candidate_indices[np.argsort(-scores[candidate_indices])]

        unique_results = []
        for idx in sorted_candidates:
            if database[idx].instance_id != current_id:
                unique_results.append(database[idx])
                if len(unique_results) == k:
                    break

        results.append(unique_results)

    return results


def main(
    input_jsonl: str,
    output_jsonl: str,
):
    instances = load_from_jsonl(input_jsonl)
    diffs = [instance.diff for instance in instances]
    embeddings = get_embeddings(diffs)
    for instance, embedding in zip(instances, embeddings):
        instance.embedding = embedding
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for instance in instances:
            f.write(instance.model_dump_json() + "\n")


if __name__ == "__main__":
    Fire(main)
