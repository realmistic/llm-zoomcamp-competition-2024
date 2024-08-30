import sys
import pandas as pd
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# OpenAI client setup
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

def llm(prompt, model):
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response

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
    problem_id = row['problem_id']
    problem_text = row['problem_text']
    problem_answer = row.get('answer')
    numerical_answer, llm_reasoning = solve_one_problem(row, model=model)
    
    correct = None
    if problem_answer is not None:
        correct = str(numerical_answer) == problem_answer
        if correct:
            print('CORRECT ANSWER')
        else:
            print('WRONG ANSWER')
            print(f' Let\'s compare answers: LLM_ANSWER: {numerical_answer}, TRUE_ANSWER: {problem_answer}')

    return {
        'problem_id': problem_id,
        'problem_text': problem_text,
        'problem_answer': problem_answer,
        'llm_reasoning': llm_reasoning,
        'llm_answer': str(numerical_answer),
        'is_correct': correct
    }

def map_progress(pool, seq, f):
    results = []
    with tqdm(total=len(seq)) as progress:
        futures = []
        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)
        for future in futures:
            result = future.result()
            results.append(result)
    return results

def main(model, dataset_type):
    if dataset_type not in ['test', 'train']:
        print("Invalid dataset type. Use 'test' or 'train'.")
        return

    df = pd.read_csv(f'input_data/{dataset_type}.csv')
    rows = df.to_dict(orient='records')

    with ThreadPoolExecutor(max_workers=14) as pool:
        results = map_progress(pool, rows, lambda row: process_row(row, model))

    df_results = pd.DataFrame(results)
    
    if dataset_type == 'train':
        df_results['is_correct_num'] = df_results.is_correct.astype(int)
        score_ratio = sum(df_results['is_correct_num']) / len(df_results)
        print(f"Score ratio: {score_ratio}")
    elif dataset_type == 'test':
        submission = df_results[['problem_id', 'llm_answer']]
        submission = submission.rename(columns={'llm_answer': 'answer'})
        submission_file = f'submission_{model}.csv'
        submission.to_csv(submission_file, index=False)
        print(f"Submission file '{submission_file}' has been created.")

    output_file = f'{dataset_type}_results_{model}.csv'
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name> <test|train>")
    else:
        model_name = sys.argv[1]
        dataset_type = sys.argv[2]
        main(model_name, dataset_type)