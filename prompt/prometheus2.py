system="""You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."""

user = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{input}

###Response A:
{output_1}

###Response B:
{output_2}

###Score Rubric:
Does the model provide relevant and useful responses to the user's needs or questions?

###Feedback: """

output_pattern = {
    1: r"\[RESULT\] ?A",
    2: r"\[RESULT\] ?B"
}