# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from fast_detect_gpt import get_sampling_discrepancy_analytic
from model import load_tokenizer, load_model


# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        with open('llama-8B-ref.json', 'r') as fin:
            res = json.load(fin)
            self.real_crits.extend(res['predictions']['real'])
            self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)

# run interactive local inference
def run(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)
    # input text
    print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
    print('')
    text = """On this page, create, edit, and delete server access configurations and server groups that give you control over interaction between PyCharm and servers. Anytime you are going to use a server, you need to define a server access configuration, no matter whether your server is on a remote host or on your computer. For more information about working with server access configurations and server groups, refer to Connect to a web server and Organize servers into groups."""

    with open('simpo_large_300_merge_latex_review') as f:
        data = json.load(f)

    data_id = [i['paperId'] for i in data]
    train_data_id = data_id[:100]

    eval_data_id = data_id[100:500]

    texts_human = {}
    texts_model = {}
    for item in data:
        context = ''
        context += item['messages'][2]['content'].split('```latex')[1].split('\end{abstract}')[0]+'\end{abstract}'
        for s,s2 in item['sections']:
            context+=r'\section{'+s+'}\n'
            context+=s2+'\n'
        texts_human[item['paperId']] = context
        texts_model[item['paperId']] = item['latex']

    score_human = []
    score_model = []
    for i in tqdm(train_data_id):
        # evaluate text
        tokenized = scoring_tokenizer(texts_human[i], return_tensors="pt", padding=True, return_token_type_ids=False,max_length=2048,truncation=True).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False,max_length=2048,truncation=True).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            crit = criterion_fn(logits_ref, logits_score, labels)
            score_human.append(crit)
        # estimate the probability of machine generated text
    for i in tqdm(train_data_id):
        # evaluate text
        tokenized = scoring_tokenizer(texts_model[i], return_tensors="pt", padding=True, return_token_type_ids=False,max_length=2048,truncation=True).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False,max_length=2048,truncation=True).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            crit = criterion_fn(logits_ref, logits_score, labels)
            score_model.append(crit)
        # estimate the probability of machine generated text

    score_human_ref = []
    score_model_ref = []
    for i in range(100):
        sh = score_human[i]
        sm = score_model[i]
        if str(sh) != 'nan' and str(sm) != 'nan':
            score_human_ref.append(sh)
            score_model_ref.append(sm)

    human_info = {}
    human_info['name'] = 'llama-8B'
    human_info["info"] = {"n_samples": 400}
    predictions = {'real': score_human_ref, 'samples': score_model_ref}
    human_info["predictions"] = predictions

    with open('./llama-8B-ref.json','w',encoding='utf-8') as f:
        json.dump(human_info,f,indent=4)

    test_score_model = []
    test_score_human = []
    for i in tqdm(eval_data_id):
        # evaluate text
        tokenized = scoring_tokenizer(texts_human[i], return_tensors="pt", padding=True, return_token_type_ids=False,max_length=2048,truncation=True).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False,max_length=2048,truncation=True).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            crit = criterion_fn(logits_ref, logits_score, labels)
            score_model.append(crit)
        # estimate the probability of machine generated text
        prob = prob_estimator.crit_to_prob(crit)
        print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be fake.')
        print()
        if prob < 0.5:
            test_score_human.append(0)
        else:
            test_score_human.append(1)
    print('Start to model')
    for i in tqdm(eval_data_id):
        # evaluate text
        tokenized = scoring_tokenizer(texts_model[i], return_tensors="pt", padding=True, return_token_type_ids=False,max_length=2048,truncation=True).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False,max_length=2048,truncation=True).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            crit = criterion_fn(logits_ref, logits_score, labels)
            score_model.append(crit)
        # estimate the probability of machine generated text
        prob = prob_estimator.crit_to_prob(crit)
        print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be fake.')
        print()
        if prob < 0.5:
            test_score_model.append(0)
        else:
            test_score_model.append(1)

    acc = []
    for i in test_score_model:
        if i == 1:
            acc.append(1)
        else:
            acc.append(0)

    for i in test_score_human:
        if i == 1:
            acc.append(0)
        else:
            acc.append(1)
    print(sum(acc)/len(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="/zhuminjun/model/LLM-Research/Meta-Llama-3___1-8B")  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default="/zhuminjun/model/LLM-Research/Meta-Llama-3___1-8B")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="/zhuminjun/LLM/exfast-detect-gpt-main/local_infer_ref")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    run(args)



