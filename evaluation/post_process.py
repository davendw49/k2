import json
import re
import numpy as np
import pickle
import math
import argparse

token_setup = {
    "llama": {
        'A': 319,
        'B': 350,
        'C': 315,
        'D': 360,
        'E': 382,
        'F': 383,
        'G': 402,
        'H': 379,
        'a': 263,
        'b': 289,
        'c': 274,
        'd': 270,
        'e': 321,
        'f': 285,
        'g': 330,
        'h': 298,
        'true': 1565,
        'false': 2089,
        'True': 5852,
        'False': 7700
    },
    "gpt2": {
        'A': 32,
        'B': 33,
        'C': 34,
        'D': 35,
        'E': 36,
        'F': 37,
        'G': 38,
        'H': 39,
        'a': 64,
        'b': 65,
        'c': 66,
        'd': 67,
        'e': 68,
        'f': 69,
        'g': 70,
        'h': 71,
        'true': 7942,
        'false': 9562,
        'True': 17821,
        'False': 25101
    },
    "mpt": {
        'A': 34,
        'B': 35,
        'C': 36,
        'D': 37,
        'E': 38,
        'F': 39,
        'G': 40,
        'H': 41,
        'a': 66,
        'b': 67,
        'c': 68,
        'd': 69,
        'e': 70,
        'f': 71,
        'g': 72,
        'h': 73,
        'true': 5672,
        'false': 7750,
        'True': 5088,
        'False': 5653
    },
    "gal": {
        'A': 55,
        'B': 56,
        'C': 57,
        'D': 58,
        'E': 59,
        'F': 60,
        'G': 61,
        'H': 62,
        'a': 87,
        'b': 88,
        'c': 89,
        'd': 90,
        'e': 91,
        'f': 92,
        'g': 93,
        'h': 94,
        'true': 11852,
        'false': 18362,
        'True': 8409,
        'False': 10394
    }
}
base_benchmark_path = r'./benchmarks/'

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-m', help='name of the model, should be unique', required=True)
parser.add_argument('--benchmark_name', '-b', help='name of the benchmark, should be unique', required=True)
args = parser.parse_args()

model = args.model_name
benchmark = args.benchmark_name

if "k2" in model or "K2" in model or "llama" in model or "vicuna" in model or "alpaca" in model:
    print('using llama tokenizer')
    token2id = token_setup['llama']
elif "mpt" in model:
    print('using mpt tokenizer')
    token2id = token_setup['mpt']
elif "gpt" in model:
    print('using gpt tokenizer')
    token2id = token_setup['gpt2']
elif "gal" in model:
    print('using gal tokenizer')
    token2id = token_setup['gal']

def extract_choice_id_ls(text: str):
    pat = r'(?<=[^\w\-])[A-Z](?=[^\w\-])'
    match = re.findall(pat, text)
    cap_ls = np.unique(sorted([ord(i)-ord('A') for i in match]))
    maximum = 0
    for i in range(0, min(len(cap_ls), 26)):
        if cap_ls[i] == i:
            maximum += 1
    return [token2id[chr(i + ord('A'))] for i in range(0, maximum)]

def process_tf(filepath, softmax):
    with open(filepath, 'r') as f:
        data = json.load(f)
    key = 'tf'
    question_ls = []
    answer_ls = []
    for i in range(len(data[key]["question"])):
        q = data[key]["question"][i]
        a = data[key]['answer'][i]
        question_ls.append(q)
        answer_ls.append(a)
    
    tf_res = []
    for i in range(len(question_ls)):
        prob_ls = []
        tf_id_ls = [token2id['True'], token2id['False']]
        for j in range(len(tf_id_ls)):
            c_id = tf_id_ls[j]
            prob_ls.append(softmax[i].tolist()[c_id])
        prob_ls_normalized = [i/(sum(prob_ls)) for i in prob_ls]
        entropy = -sum([i * math.log(i,2) for i in prob_ls_normalized])
        tf_res.append({
            'answer': answer_ls[i],
            'probability_distribution': prob_ls,
            'normalized': prob_ls_normalized,
            'normalized_entropy': entropy,
        })

    correct = 0
    total = 0
    for i in tf_res:
        ans = 0 if i['answer'] == 'True' else 1
        total += 1
        norm = np.array(i['normalized'])
        if ans == norm.argmax():
            correct += 1
    return tf_res, correct / total

def process_choice(filepath, softmax):
    with open(filepath, 'r') as f:
        data = json.load(f)
    if 'npee' in filepath or 'kaoyan' in filepath or 're' in filepath:
        key = 'choice'
        question_ls = []
        answer_ls = []
        for i in range(len(data[key]["question"])):
            q = data[key]["question"][i]
            a = data[key]['answer'][i]
            question_ls.append(q)
            answer_ls.append(a)
        
        choice_res = []
        choice_num_ls = []
        for i in range(len(question_ls)):
            prob_ls = []
            question = question_ls[i]
            choice_id_ls = extract_choice_id_ls(question) # capital letters only
            choice_num_ls.append(len(choice_id_ls))
            for j in range(len(choice_id_ls)):
                c_id = choice_id_ls[j]
                prob_ls.append(softmax[i].tolist()[c_id])
            prob_ls_normalized = [i/(sum(prob_ls)) for i in prob_ls]
            entropy = -sum([i * math.log(i,2) for i in prob_ls_normalized])
            choice_res.append({
                'answer': answer_ls[i],
                'probability_distribution': prob_ls,
                'normalized': prob_ls_normalized,
                'normalized_entropy': entropy,
            })
    else:
        question_ls = []
        answer_ls = []
        for item in data:
            q = item["question"]
            a = item['answerKey']
            question_ls.append(q)
            answer_ls.append(a)
        
        choice_res = []
        choice_num_ls = []
        for i in range(len(question_ls)):
            prob_ls = []
            question = question_ls[i]
            choice_id_ls = [token2id[cho['label']] for cho in question['choices']] # capital letters only
            choice_num_ls.append(len(choice_id_ls))
            for j in range(len(choice_id_ls)):
                c_id = choice_id_ls[j]
                prob_ls.append(softmax[i].tolist()[c_id])
            prob_ls_normalized = [i/(sum(prob_ls)) for i in prob_ls]
            entropy = -sum([i * math.log(i,2) for i in prob_ls_normalized])
            choice_res.append({
                'answer': answer_ls[i],
                'probability_distribution': prob_ls,
                'normalized': prob_ls_normalized,
                'normalized_entropy': entropy,
            })
        

    correct = 0
    random_correct_expectation = 0
    total = 0
    for i in range(len(choice_res)):
        try:
            ans = ord(choice_res[i]['answer']) - ord('A')
        except:
            continue
        total += 1
        norm = np.array(choice_res[i]['normalized'])
        if ans == norm.argmax():
            correct += 1
        random_correct_expectation += 1 / choice_num_ls[i]
    return choice_res, correct / total, random_correct_expectation / total


print('Start to evaluate ...')
with open(f'./pickle/softmax_{model}_{benchmark}.pickle', 'rb') as f:
    softmax = pickle.load(f)

if 'npee' in benchmark or 'kaoyan' in benchmark:
    choice_softmax_wa = softmax['choice']['wa'] if 'choice' in softmax else []
    choice_softmax_woa = softmax['choice']['woa'] if 'choice' in softmax else []
    choice_res_wa, correct_rate_wa, correct_rate_random = process_choice(f'{base_benchmark_path}benchmark_{benchmark}.json', choice_softmax_wa)
    choice_res_woa, correct_rate_woa, correct_rate_random = process_choice(f'{base_benchmark_path}benchmark_{benchmark}.json', choice_softmax_woa)
    with open(f"./results/{model.replace('-', '_')}_result_on_{benchmark}.txt", 'w') as f:
        f.write(f'Correct rate (choice, with "the answer is ", {model}) is {correct_rate_wa}' + '\n')
        f.write(f'Correct rate (choice, without "the answer is ", {model}) is {correct_rate_woa}' + '\n')
        f.write(f'Correct rate (choice, random) is {correct_rate_random}' + '\n')

print('Finished!')