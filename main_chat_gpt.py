import sys
import pandas as pd
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import json
import os

from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file


# CHANGES VS main.py, that was crashing on API timeout after some time 
# To address this issue and improve the script's reliability, you could consider the following modifications:

# Implement error handling and retries for API calls.
# Increase the timeout for API requests.
# Implement checkpointing to save progress periodically, allowing you to resume from where it left off in case of errors.
# Consider processing the data in smaller batches to reduce the impact of individual timeouts.

# GPT-4o-mini client setup with increased timeout
client = OpenAI()
    # base_url='https://api.openai.com/v1/',
    # api_key=os.getenv("OPENAI_API_KEY"),
    # timeout=5 * 60.0  # Increase timeout to 5 minutes
# )

MAX_RETRIES = 3
RETRY_DELAY = 5
BATCH_SIZE = 5

def llm(prompt, model):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                timeout=5*60,
                messages=[{"role": "user", "content": prompt}]
            )
            return response
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Attempt {attempt + 1} failed. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                raise e

def get_answer(question, model):
    prompt = f"""Role:
    You are an advanced AI system with exceptional mathematical reasoning and problem-solving capabilities, specifically designed to solve tricky math problems (whose answer is a non-negative integer) written in LaTeX format from the AI Mathematical Olympiad (AIMO) competition. Your task is to accurately analyze and solve intricate mathematical problems, demonstrating a deep understanding of mathematical concepts and a strong ability to apply logical reasoning strategies.

    Instruction:
    1. Carefully read and comprehend the problem statement provided in the "Problem" section.
    2. In the "Solution" section, provide a solution of the problem with detailed explanation of your logical reasoning process. Keep in mind that answer must be a non-negative integer number.
    3. At the end, create a "Answer" section where you will state only the final numerical (convert fractions to approx. numbers, try not to do rounding) or algebraic answer, without any additional text or narrative.

    Problem:
    ...

    Solution:
    ...

    Answer:
    ...

    {question}

    Step-by-step solution and final answer:"""

    response = llm(prompt=prompt, model=model)
    return response

def extract_numerical_answer(text):
    match = re.search(r'(?:final answer|the answer is)[:\s]*([+-]?\d*\.?\d+)', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        numbers = re.findall(r'[+-]?\d*\.?\d+', text)
        if numbers:
            number = float(numbers[-1])
            if number.is_integer():
                return int(number)
            else:
                return number
    return None

def solve_one_problem(one_row, model):
    llm_response = get_answer(question=one_row['problem_text'], model=model)
    ans = llm_response.choices[0].message.content
    res = extract_numerical_answer(ans)
    return res, llm_response.choices[0].message.content

def process_row(row, model):
    try:
        problem_id = row['problem_id']
        problem_text = row['problem_text']
        problem_answer = row.get('answer')
        numerical_answer, llm_reasoning = solve_one_problem(row, model=model)
        
        correct = None
        if problem_answer is not None:
            correct = str(numerical_answer) == problem_answer
            print('CORRECT ANSWER' if correct else 'WRONG ANSWER')
            if not correct:
                print(f' Let\'s compare answers: LLM_ANSWER: {numerical_answer}, TRUE_ANSWER: {problem_answer}')

        return {
            'problem_id': problem_id,
            'problem_text': problem_text,
            'problem_answer': problem_answer,
            'llm_reasoning': llm_reasoning,
            'llm_answer': str(numerical_answer),
            'is_correct': correct
        }
    except Exception as e:
        print(f"Error processing row {row['problem_id']}: {str(e)}")
        return None

def process_batch(batch, model):
    return [process_row(row, model) for row in batch if row is not None]

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'processed': 0, 'results': []}

def save_checkpoint(checkpoint_file, processed, results):
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed': processed, 'results': results}, f)

def main(model, dataset_type):
    if dataset_type not in ['test', 'train']:
        print("Invalid dataset type. Use 'test' or 'train'.")
        return

    df = pd.read_csv(f'input_data/{dataset_type}.csv')
    rows = df.to_dict(orient='records')

    checkpoint_file = f'{dataset_type}_checkpoint_{model}.json'
    checkpoint = load_checkpoint(checkpoint_file)
    start_index = checkpoint['processed']
    results = checkpoint['results']
    print(f'Start index (if checkpoint loaded ==> it will be >0): {start_index}')

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
        for i in tqdm(range(start_index, len(rows), BATCH_SIZE), initial=start_index, total=len(rows)):
            batch = rows[i:i+BATCH_SIZE]
            batch_results = list(pool.map(lambda row: process_row(row, model), batch))
            results.extend([r for r in batch_results if r is not None])
            save_checkpoint(checkpoint_file, i + len(batch), results)

    df_results = pd.DataFrame(results)
    
    if dataset_type == 'train':
        df_results['is_correct_num'] = df_results.is_correct.astype(int)
        score_ratio = df_results['is_correct_num'].mean()
        print(f"Score ratio: {score_ratio}")
    
    elif dataset_type == 'test':
        submission = df_results[['problem_id', 'llm_answer']]
        submission = submission.rename(columns={'llm_answer': 'answer'})
        submission_file = f'submission_v2{model}.csv'
        submission.to_csv(submission_file, index=False)
        print(f"Submission file '{submission_file}' has been created.")

    output_file = f'{dataset_type}_results_v2_{model}.csv'
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name> <test|train>")
    else:
        model_name = sys.argv[1]
        dataset_type = sys.argv[2]
        main(model_name, dataset_type)
