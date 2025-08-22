import json

from fire import Fire

from commit import load_from_jsonl
from commit.retriever.embedding import retrieve

DIRECT_PROMPT = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit.
## Input Format:
--- START OF CODE DIFF ---
(Code changes in .diff format)
--- END OF CODE DIFF ---

## Output Format:
A concise commit message describing the code changes, wrapped in <message> </message> tags.
E.g. <message>Fixed a bug in the user authentication flow</message>
E.g. <message>feat(server): Add new API endpoint for user registration</message>
"""

SIMILAR_PROMPT = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit. First, a similar commit example (including both code diff and commit message) is provided for reference. Then, you will be given a code diff which is your task, and you need to write a commit message for it.

## Input Format:

=== START OF SIMILAR COMMIT ===

--- START OF CODE DIFF ---
(Code changes in .diff format)
--- END OF CODE DIFF ---

--- START OF COMMIT MESSAGE ---
A commit message describing the code changes, wrapped in <message> </message> tags.
E.g. <message>Fixed a bug in the user authentication flow</message>
--- END OF COMMIT MESSAGE ---

=== END OF SIMILAR COMMIT ===

=== START OF YOUR TASK ===

--- START OF CODE DIFF ---
(Code changes in .diff format)
--- END OF CODE DIFF ---

=== END OF YOUR TASK ===

## Output Format:

A concise commit message describing the code changes, wrapped in <message> </message> tags.
E.g. <message>Fixed a bug in the user authentication flow</message>
E.g. <message>feat(server): Add new API endpoint for user registration</message>
"""


prompt_mapping = {
    "default": DIRECT_PROMPT,
    "similar": SIMILAR_PROMPT,
}


def make_tasks(
    dataset_path: str,
    tasks_path: str,
    prompt_type: str = "default",
    database_path: str = None,
):
    dataset = load_from_jsonl(dataset_path)

    if prompt_type == "similar" and not database_path:
        raise ValueError(
            "For 'similar' prompt type, please provide a database path to retrieve similar commits."
        )

    if database_path:
        database = load_from_jsonl(database_path)
        examples = retrieve(dataset, database)

    tasks = []
    for i, instance in enumerate(dataset):
        system_prompt = prompt_mapping[prompt_type]
        user_prompt = ""
        task_id = f"{instance.owner}_{instance.repo}_{instance.commit_sha[:7]}"

        if prompt_type == "similar":
            for example in examples[i]:
                user_prompt += (
                    f"=== START OF SIMILAR COMMIT ===\n"
                    f"--- START OF CODE DIFF ---\n{example.diff}\n--- END OF CODE DIFF ---\n"
                    f"--- START OF COMMIT MESSAGE ---\n<message>{example.message}</message>\n"
                    f"--- END OF COMMIT MESSAGE ---\n"
                    f"=== END OF SIMILAR COMMIT ===\n\n"
                )

        if prompt_type == "default":
            user_prompt += (
                f"--- START OF CODE DIFF ---\n{instance.diff}\n--- END OF CODE DIFF ---"
            )

        if prompt_type == "similar":
            user_prompt += (
                f"\n\n=== START OF YOUR TASK ===\n"
                f"--- START OF CODE DIFF ---\n{instance.diff}\n--- END OF CODE DIFF ---\n"
                f"=== END OF YOUR TASK ==="
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tasks.append(
            {
                "task_id": task_id,
                "messages": messages,
                "label": instance.message,
            }
        )

    with open(tasks_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


if __name__ == "__main__":
    Fire(make_tasks)
