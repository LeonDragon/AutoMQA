# List of prompts for Gemini experiments
PROMPTS = {
    'default': {
        'answer_key': (
            "Extract the answer key from the provided image. There are total 60 questions for each test code. "
            "Return results in JSON format with exam title, code, and answer keys."
        ),
        'column_analysis': (
            "You are an expert at reading and interpreting multiple choice answer sheets. "
            "Carefully analyze the provided image of an answer sheet column and extract the selected answers. "
            "Follow these guidelines:\n"
            "1. Each question has 4 bubbles labeled A, B, C, D from left to right\n"
            "2. Look for any marks in the bubbles - they could be fully filled, partially filled, or lightly shaded\n"
            "3. If multiple bubbles are marked for a question, choose the darkest/most filled one\n"
            "4. If no bubbles are marked or the marks are too faint, return 'X' for that question\n"
            "5. Return results in JSON format with question numbers as keys and answers as values"
        ),
        'recheck_analysis': (
            "You are analyzing an answer sheet column to provide detailed reasons for uncertain answers. Please:\n"
            "1. Carefully examine each question's bubbles\n"
            "2. For questions marked with X (uncertain), provide a detailed reason explaining why\n"
            "3. Return results in JSON format with question numbers as keys and reasons as values\n"
            "4. Only include questions marked with X in the response\n"
            "5. Be as specific as possible in your reasoning, describing exactly what you observe\n"
            "6. Do not attempt to guess or infer the correct answer\n"
            "7. If you notice any patterns or anomalies across multiple questions, note them in your reasoning"
        )
    },
    'experiment_1': {
        'answer_key': (
            "Analyze the answer key image carefully. Extract all test codes and their corresponding answers. "
            "Return results in JSON format with exam details and answer keys."
        ),
        'column_analysis': (
            "Analyze this answer sheet column image. For each question, identify the selected answer (A, B, C, D) "
            "or mark as 'X' if uncertain. Return results in JSON format."
        )
    },
    'experiment_2': {
        'answer_key': (
            "Process the answer key image. Extract all test codes and answers. "
            "Return structured JSON data with exam information and answer keys."
        ),
        'column_analysis': (
            "Process this answer sheet column. Identify selected answers or mark uncertain ones. "
            "Return JSON data with question numbers and answers."
        )
    }
}

def get_prompt(experiment='default', prompt_type='column_analysis'):
    """Get a prompt from the PROMPTS dictionary"""
    return PROMPTS.get(experiment, {}).get(prompt_type, '')
