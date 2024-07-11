user = """[INST] You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {input}
***
[Response 1]: {output_1}
***
[Response 2]: {output_2}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided. [/INST]"""

output_pattern = {
    1: r"final decision is (R|r)esponse 1",
    2: r"final decision is (R|r)esponse 2",
    3: r"final decision is (T|t)ie"
}