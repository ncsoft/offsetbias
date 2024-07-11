user = """Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.

### Instruction:
{input}

### Response 1:
{output_1}

### Response 2:
{output_2}

### Evaluation:"""

output_pattern = {
    1: r"^1\n\n",
    2: r"^2\n\n",
    3: r"^Tie\n\n"
}