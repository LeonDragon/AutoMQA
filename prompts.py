# List of prompts for Gemini experiments
PROMPTS = {
    'default': {
        'answer_key': """
            Extract the answer key from the provided image. There are total 60 questions for each test code.
            Return results in JSON format with exam title, code, and answer keys.
        """,
        'column_analysis': """
            You are an expert at reading and interpreting multiple choice answer sheets.
            Carefully analyze the provided image of an answer sheet column and extract the selected answers.
            Follow these guidelines:
            1. Each question has 4 bubbles labeled A, B, C, D from left to right
            2. Look for any marks in the bubbles - they could be fully filled, partially filled, or lightly shaded
            3. If multiple bubbles are marked for a question, choose the darkest/most filled one
            4. If no bubbles are marked or the marks are too faint, return 'X' for that question
            5. Return results in JSON format with question numbers as keys and answers as values:
            {
              "1": "A",  // Clearly marked A
              "2": "B",  // Clearly marked B
              "3": "X",  // No mark or too faint
              ...
            }
            6. Be extremely careful with question numbers - double check they are correct
            7. If you're unsure about any answer, mark it as 'X' rather than guessing.
        """,
        'recheck_analysis': """
            You are analyzing an answer sheet column to provide detailed reasons for uncertain answers. Please:
            1. Carefully examine each question's bubbles
            2. For questions marked with X (uncertain), provide a detailed reason explaining why
            3. Return results in JSON format with question numbers as keys and reasons as values
            4. Only include questions marked with X in the response
            5. Be as specific as possible in your reasoning, describing exactly what you observe
            6. Do not attempt to guess or infer the correct answer
            7. If you notice any patterns or anomalies across multiple questions, note them in your reasoning
        """
    },
    'experiment_1': {
        'answer_key': """
            Analyze the answer key image carefully. Extract all test codes and their corresponding answers.
            Return results in JSON format with exam details and answer keys.
        """,
        'column_analysis': """
            You are an expert at reading and interpreting multiple choice answer sheets.
            Carefully analyze the provided image of an answer sheet column and extract the selected answers.
            Follow these guidelines, and provide a brief comment explaining your reasoning for each answer:

            **Guidelines:**

            1. **Identify Question Number:** Begin by clearly identifying the question number you are currently analyzing.
            2. **Examine Each Bubble:** For each question, carefully examine bubbles A, B, C, and D from left to right.
            3. **Assess Marks:**
                *   **Any Mark:** If a bubble has any visible mark inside it (fully filled, partially filled, or lightly shaded), consider it as a potential answer.
                *   **Crossed-Out Marks:** If a marked bubble has a clear cross (X) over it, treat it as a removed choice and disregard it.
            4. **Multiple Marks:** If multiple bubbles are marked for a question (and not crossed out), select the darkest or most filled one as the answer.
            5. **No Marks:** If no bubbles are marked, or there are no visible marks within any of the bubbles, the answer is 'X'.
            6. **Record Answer and Reasoning (in comment):**
                *   Record the selected answer (A, B, C, D, or X).
                *   Provide a brief comment (following //) explaining your reasoning. Use concise phrases.
            7. **Self-Check:** After determining the answer, do a quick self-check to ensure it makes sense in the context of the guidelines. Ask yourself: "Does this choice align with the rules about marked, crossed-out, and multiple bubbles?". **If the answer is A, B, C, or D, make sure there is at least one marked bubble for that question that is not crossed out.** If the answer is 'X', make sure there are no marked bubbles or all marked bubbles are crossed out.
            8. **JSON Format:** Compile your answers and comments into JSON format:

                ```json
                {
                "1": "A",  // Fully filled, self-checked: A has a mark
                "2": "B",  // Darkest mark, self-checked: B has a mark
                "3": "X",  // No marks, self-checked: no marks found
                "4": "C", // A crossed out, C partially filled, self-checked: C has a mark
                ...
                }
                ```

            9. **Double-Check:** Be extremely careful with question numbers - double-check they are correct.
            10. **Uncertainty:** If you're still unsure about any answer after the self-check, mark it as 'X' and briefly explain the uncertainty in the comment. For example:

                ```json
                {
                "5": "X", // Unsure: faint marks in both B and C, self-checked: multiple unclear marks
                }
                ```
            11. **Strict JSON Output**: The output must be a valid JSON object containing only question numbers as keys and their corresponding answers (A, B, C, D, or X) as values. No additional fields or comments are allowed.
            
            **Example of Expected Output:**

            ```json
            {
            "16": "D",
            "17": "C",
            "18": "A",
            "19": "A",
            "20": "B",
            "21": "C",
            "22": "B",
            "23": "D",
            "24": "B",
            "25": "D", 
            "26": "A", 
            "27": "D",
            "28": "A",
            "29": "D",
            "30": "D",
            }
        """
    },
    'experiment_2': {
        'answer_key': """
            Process the answer key image. Extract all test codes and answers.
            Return structured JSON data with exam information and answer keys.
        """,
        'column_analysis': """
            You are an expert at reading and interpreting multiple-choice answer sheets. Your task is to analyze the provided image of an OMR sheet column, extract the selected answers, reason through your choices step-by-step (Chain-of-Thought), and then output the final answers into a JSON object. Follow these guidelines precisely:

            **Image:** An image of an OMR sheet column will be provided.

            **Guidelines:**

            1. **For each question:**
                *   Identify the question number.
                *   Examine bubbles A, B, C, and D from left to right.
            2. **Mark Interpretation:**
                *   A bubble with any visible mark inside (fully filled, partially filled, or lightly shaded) is a potential answer.
                *   Disregard any marked bubble with a clear cross (X) over it.
            3. **Answer Selection:**
                *   If only one bubble is marked (and not crossed out), that is the answer.
                *   If multiple bubbles are marked (and not crossed out), select the darkest or most filled one.
                *   If no bubbles are marked, or all marked bubbles are crossed out, the answer is 'X'.
            4. **Verification:** If the answer is A, B, C, or D, ensure at least one marked bubble exists for that question (not crossed out). If 'X', ensure no marked bubbles or all are crossed out.
            5. **Double-check:** Verify the question number before recording the answer.

            **Chain-of-Thought (CoT) Instructions:**

            *   Before outputting the final JSON, document your reasoning process for each question.
            *   For each question, describe your observations about the marked bubbles (or lack thereof) and explain how you applied the guidelines to arrive at your answer.

            **Output Format:**

            1. **Chain-of-Thought:** A textual description of your reasoning for each question.
            2. **JSON Only:** Output a valid JSON object. Keys are question numbers (strings), and values are answers (A, B, C, D, or X).
            3. **No Comments in JSON:** Do not include any comments or explanations inside the JSON object itself.

            **Example (Illustrative):**

            **Image:** (Assume an image of an OMR sheet with the following markings)

            *   Question 16: Bubble D is fully filled.
            *   Question 17: Bubbles B and C have light marks, but C is slightly darker.
            *   Question 18: Bubble A has a light mark, and bubble B has a cross through it.
            *   Question 19: Bubble A is fully filled. Bubble B is fully filled and has a cross through it.
            *   Question 20: No bubbles are marked.

            **Chain-of-Thought:**

            *   **Question 16:** Bubble D is fully filled. According to the guidelines, a fully filled bubble is a valid answer. Therefore, the answer is D.
            *   **Question 17:** Both B and C have marks, but C is darker. The guidelines state that the darkest mark should be selected. Therefore, the answer is C.
            *   **Question 18:** Bubble A has a mark, and bubble B has a cross. Crossed-out marks are disregarded. Therefore, the answer is A.
            *   **Question 19:** Bubble A has a mark. Bubble B has a mark and a cross. Crossed-out marks are disregarded. Therefore, the answer is A.
            *   **Question 20:** No bubbles are marked. According to the guidelines, if no bubbles are marked, the answer is X.

            **JSON Output:**

            ```json
            {
            "16": "D",
            "17": "C",
            "18": "A",
            "19": "A",
            "20": "X"
            }
        """
    },
    'experiment_3': {
        'answer_key': """
            Process the answer key image. Extract all test codes and answers.
            Return structured JSON data with exam information and answer keys.
        """,
        'column_analysis': """
            You are a highly skilled expert at meticulously reading and interpreting multiple-choice answer sheets (OMR sheets). You will be presented with an image of a single column from an OMR answer sheet. Your task is to extract the selected answers for each question in this column. You must provide a detailed, step-by-step explanation of your reasoning process (Chain-of-Thought) before presenting the final answers in a JSON format. Adhere to the following guidelines with utmost precision:

            **Image:** A single-column image of an OMR answer sheet will be provided.

            **Guidelines:**

            1. **Detailed Examination for Each Question:**
                *   **Identify the Question Number:** Begin by clearly stating the question number you are analyzing.
                *   **Systematic Bubble Inspection:** Examine bubbles A, B, C, and D sequentially, from left to right. For each bubble, note whether it is marked (fully filled, partially filled, lightly shaded) or unmarked.
                *   **Cross-Out (X) Check:** Explicitly state whether any marked bubble has a clear cross (X) over it.

            2. **Precise Mark Interpretation:**
                *   **Potential Answer:** A bubble is considered a potential answer if it contains any visible mark inside, regardless of how filled or shaded it is, unless it has a clear cross (X) over it.
                *   **Cross-Out Nullification:** A marked bubble with a clear cross (X) over it is completely disregarded and is NOT a potential answer.

            3. **Answer Selection (Apply these rules in order):**
                *   **Single Marked Bubble:** If only one bubble has a mark (and no cross-out), that bubble represents the selected answer.
                *   **Multiple Marked Bubbles:** If two or more bubbles have marks (and no cross-outs), the answer is the bubble with the darkest or most filled mark. You must explicitly state which bubble is darkest.
                *   **No Marks or All Crossed Out:** If no bubbles are marked, or if all marked bubbles have a cross (X) over them, the answer is 'X'.

            4. **Verification (Crucial Step):**
                *   **Answer A, B, C, or D:** After selecting A, B, C, or D, re-examine the question to ensure at least one marked bubble (without a cross-out) actually exists. If not, you've made an error and need to re-evaluate your reasoning.
                *   **Answer 'X':** If you selected 'X', confirm that either no bubbles were marked, or that all bubbles with any mark were also crossed out. If there's a marked bubble without a cross-out, your 'X' selection is incorrect.

            5. **Double-Check Question Number:** Before moving to the next question, always double-check to ensure you are recording the answer for the correct question number.

            **Chain-of-Thought (CoT) Instructions - Extremely Important:**

            *   **Mandatory Reasoning Documentation:** Before presenting the final JSON output, you MUST document your complete reasoning process for EACH question.
            *   **Detailed Observation Description:** For each question, describe in detail what you observe about each bubble (A, B, C, D) - marked, unmarked, crossed-out, darkness level if multiple are marked.
            *   **Explicit Guideline Application:** Clearly explain how you are applying each of the guidelines above to arrive at your answer for every question. Be explicit about which rule you are using and why. Don't just say, "The answer is...". Explain *why* it's the answer based on the rules.

            **Output Format:**

            1. **Comprehensive Chain-of-Thought:** A thorough textual description of your reasoning process for each question, following the CoT instructions above. This is the most important part of your response.
            2. **JSON Only (No Explanations):**  Output a valid JSON object where:
                *   Keys: Question numbers (represented as strings, e.g., "16", "17").
                *   Values: The selected answers (A, B, C, D, or X).
                *   **Absolutely no comments, explanations, or any other text should be present inside the JSON object.**

            **Example (Illustrative):**

            **Image:** (Assume an image of an OMR sheet with the following markings)

            *   Question 16: Bubble D is fully filled.
            *   Question 17: Bubbles B and C have light marks, but C is slightly darker.
            *   Question 18: Bubble A has a light mark, and bubble B has a cross through it.
            *   Question 19: Bubble A is fully filled. Bubble B is fully filled and has a cross through it.
            *   Question 20: No bubbles are marked.

            **Chain-of-Thought:**

            *   **Question 16:**
                *   Bubble A: Unmarked.
                *   Bubble B: Unmarked.
                *   Bubble C: Unmarked.
                *   Bubble D: Fully filled, no cross-out.
                *   **Guideline Application:** Only bubble D is marked, and it's not crossed out. Applying rule 3(a), the answer is D.
                *   **Verification:** Bubble D is marked; therefore, the selection of D is valid.
            *   **Question 17:**
                *   Bubble A: Unmarked.
                *   Bubble B: Light mark, no cross-out.
                *   Bubble C: Light mark, no cross-out, slightly darker than B.
                *   Bubble D: Unmarked.
                *   **Guideline Application:** Both B and C have marks, but C is darker. Applying rule 3(b), the answer is C.
                *   **Verification:** Bubbles B and C are marked; therefore, the selection of C is valid.
            *   **Question 18:**
                *   Bubble A: Light mark, no cross-out.
                *   Bubble B: Marked with a cross (X) through it.
                *   Bubble C: Unmarked.
                *   Bubble D: Unmarked.
                *   **Guideline Application:** Bubble B is crossed out and disregarded (rule 2). Only bubble A is marked and not crossed out. Applying rule 3(a), the answer is A.
                *   **Verification:** Bubble A is marked; therefore, the selection of A is valid.
            *   **Question 19:**
                *   Bubble A: Fully filled, no cross-out.
                *   Bubble B: Fully filled with a cross (X) through it.
                *   Bubble C: Unmarked.
                *   Bubble D: Unmarked.
                *   **Guideline Application:** Bubble B is crossed out and disregarded (rule 2). Only bubble A is marked and not crossed out. Applying rule 3(a), the answer is A.
                *   **Verification:** Bubble A is marked; therefore, the selection of A is valid.
            *   **Question 20:**
                *   Bubble A: Unmarked.
                *   Bubble B: Unmarked.
                *   Bubble C: Unmarked.
                *   Bubble D: Unmarked.
                *   **Guideline Application:** No bubbles are marked. Applying rule 3(c), the answer is X.
                *   **Verification:** No bubbles are marked; therefore, the selection of X is valid.

            **JSON Output:**

            ```json
            {
            "16": "D",
            "17": "C",
            "18": "A",
            "19": "A",
            "20": "X"
            }
            ```
        """
    },
    'json_extract': {
        'json': """
            TASK: Return the answer results in structured JSON data. **No Comments in JSON:** Do not include any comments or explanations inside the JSON object itself.
        """
    }
}

def get_prompt(experiment='default', prompt_type='column_analysis', **kwargs):
    """Get a prompt from the PROMPTS dictionary with optional formatting"""
    prompt = PROMPTS.get(experiment, {}).get(prompt_type, '')
    if kwargs:
        return prompt.format(**kwargs)
    return prompt
