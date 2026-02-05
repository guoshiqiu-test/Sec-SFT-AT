# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from .adapt_think_rm import adapt_think_rm
import torch
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


import re


def has_reasoning(text: str) -> bool:
    """判断是否包含解释性语句或关键词"""
    if not text or len(text.strip()) == 0:
        return False

    # 解释性关键词（可扩展）
    reasoning_keywords = [
        r'\bbecause\b',
        r'\bdue to\b',
        r'\bthis is (why|because)\b',
        r'\bthe reason (is|why)\b',
        r'\bbased on\b',
        r'\bwe can see that\b',
        r'\bfrom this\b',
        r'\btherefore\b',
        r'\bso\b',
        r'\bas a result\b',
        r'\bit is evident that\b'
    ]

    # 逻辑句型判断（防止只是重复）
    reasoning_pattern = re.compile("|".join(reasoning_keywords), re.IGNORECASE)
    return bool(reasoning_pattern.search(text))


class AdaptThinkRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', is_training=True,
                 nothinking_bonus=0, ref_result_file=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or adapt_think_rm
        self.reward_fn_key = reward_fn_key
        self.is_training = is_training
        self.nothinking_bonus = nothinking_bonus
        if ref_result_file:
            self.problem2ref_metrics = {js['problem'].strip(): js['metrics'] for js in
                                        tqdm(json.load(open(ref_result_file, 'r')), desc='LOADING REF METRICS')}
        if self.is_training:
            print(f'\n\nNOTHINKING BONUS: {self.nothinking_bonus}\n\n')

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        id2scores = {'nothinking': defaultdict(list), 'thinking': defaultdict(list)}
        id2mean_acc = defaultdict(dict)
        id2mean_len = defaultdict(dict)
        id2std_len = defaultdict(dict)
        all_scores = []
        uid2ref_metrics = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum().item()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            has_explanation = has_reasoning(response_str)
            score['has_explanation'] = has_explanation

            uid = data_item.non_tensor_batch['uid'] if self.is_training else 'validate'
            enforce_nothinking = data_item.batch['enforce_nothinking'].item()
            is_nothinking = response_str.strip().startswith('</think>')
            score.update({
                'response_length': valid_response_length,
                'ground_truth': str(ground_truth),
                'enforce_nothinking': enforce_nothinking,
                'enforce_nothinking': enforce_nothinking,
                'is_nothinking': is_nothinking,
            })
            if self.is_training:
                if enforce_nothinking:
                    score.update({
                        'nothinking_response_length': valid_response_length,
                        'nothinking_acc': score['acc'],
                        'thinking_response_length': None,
                        'thinking_acc': None,
                    })
                    # if has_explanation:
                    #     print(f"Explain example: {response_str}")
                else:
                    score.update({
                        'nothinking_response_length': None,
                        'nothinking_acc': None,
                        'thinking_response_length': valid_response_length,
                        'thinking_acc': score['acc'],
                    })
                problem = prompt_str.split('<｜User｜>')[1].split('<｜Assistant｜>')[0].strip()
                assert problem in self.problem2ref_metrics, problem
                uid2ref_metrics[uid] = self.problem2ref_metrics[problem]
            else:
                if is_nothinking:
                    score.update({
                        'nothinking_response_length': valid_response_length,
                        'nothinking_acc': score['acc'],
                        'thinking_response_length': None,
                        'thinking_acc': None,
                    })
                else:
                    score.update({
                        'nothinking_response_length': None,
                        'nothinking_acc': None,
                        'thinking_response_length': valid_response_length,
                        'thinking_acc': score['acc'],
                    })

            all_scores.append(score)
            if self.is_training:
                if score['enforce_nothinking']:
                    id2scores['nothinking'][uid].append(score)
                else:
                    id2scores['thinking'][uid].append(score)

            print_key = f"source_{data_source}_{'nothinking' if is_nothinking else 'thinking'}"
            if print_key not in already_print_data_sources:
                already_print_data_sources[print_key] = 0

            if already_print_data_sources[print_key] < self.num_examine:
                already_print_data_sources[print_key] += 1
                print(f'\n\n[data_source]{print_key}')
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for key, value in score.items():
                    print(f"[{key}]", value)

        if self.is_training:
            for key in ['nothinking', 'thinking']:
                for uid, scores in id2scores[key].items():
                    if scores:
                        id2mean_acc[key][uid] = np.mean([score['acc'] for score in scores])
                        lens = [score['response_length'] for score in scores if score['acc'] == 1]
                        if len(lens) == 0:
                            id2mean_len[key][uid] = 0
                            id2std_len[key][uid] = 1
                        else:
                            id2mean_len[key][uid] = np.mean(lens)
                            id2std_len[key][uid] = np.std(lens) + 1e-7
                print(f"id2mean_acc_{key}: {id2mean_acc}")
                print(f"id2mean_len_{key}: {id2mean_len}")

        try:
            # ✅ 计算 enforce_nothinking 的样本中，带解释的总数
            explanation_rate = np.mean([
                s['has_explanation'] for s in all_scores if s.get('enforce_nothinking', False)
            ])
            print(f"[ExplainRate] 当前 batch 中 enforce_nothinking 的解释率：{explanation_rate:.2f}")

            # ✅ 计算 enforce_nothinking 且 带解释 且 acc==1 的数量
            explanation_and_correct = sum([
                s['has_explanation'] and s['acc'] == 1 for s in all_scores if s.get('enforce_nothinking', False)
            ])
            explanation_total = sum([
                s['has_explanation'] for s in all_scores if s.get('enforce_nothinking', False)
            ])
            if explanation_total > 0:
                explanation_acc_rate = explanation_and_correct / explanation_total
            else:
                explanation_acc_rate = 0.0

            print(f"[Explain+Acc] enforce_nothinking 模式下，有解释的回答中正确率为：{explanation_acc_rate:.2f}")
            print(f"[Explain+Acc] 数量统计：正确且有解释 = {explanation_and_correct}, 总带解释 = {explanation_total}")
        except Exception as e:
            print(f"[ExplainRate] 计算失败：{e}")

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            score = all_scores[i]
            if self.is_training:
                uid = data_item.non_tensor_batch['uid']
                enforce_nothinking, acc, response_len = score['enforce_nothinking'], score['acc'], score[
                    'response_length']
                has_explanation = score.get('has_explanation', False)
                mean_acc_nothinking = id2mean_acc['nothinking'][uid]
                mean_len_nothinking, std_len_nothinking = id2mean_len['nothinking'][uid], id2std_len['nothinking'][uid]
                mean_acc_thinking = id2mean_acc['thinking'][uid]
                mean_len_thinking, std_len_thinking = id2mean_len['thinking'][uid], id2std_len['thinking'][uid]

                ref_metrics = uid2ref_metrics[uid]
                ref_mean_acc_thinking = ref_metrics['avg_acc_thinking']

                # if enforce_nothinking:
                #     reward = acc - ref_mean_acc_thinking + self.nothinking_bonus
                # else:
                #     reward = acc - ref_mean_acc_thinking

                if enforce_nothinking:
                    if acc == 1:
                        if 100 < response_len <= 200:
                            reward = 1.2 + self.nothinking_bonus  # 正确 + 简洁解释 ✅✅
                        elif response_len > 200:
                            reward = 1.0  # 正确 + 详细解释 ✅
                        else:
                            reward = 0.2  # 正确 + 太简单解释 ⚠️
                    else:
                        reward = acc - ref_mean_acc_thinking  # 错误 ❌❌
                        # if has_explanation:
                        #     reward += self.nothinking_bonus  # 即使错误也要有加一个解释奖励

                else:
                    reward = acc - ref_mean_acc_thinking
                    # === 思考模式下长度奖励 ===
                    if acc == 1:
                        if 300 <= response_len <= 2000:
                            reward = 1.2  # 理想长度 ✅
                        elif response_len < 300:
                            reward = 0.2  # 太短 ❌
                        elif response_len > 2000:
                            reward = 1.0  # 太长 ⚠️

                score['score'] = reward
                if enforce_nothinking:
                    score.update({
                        "reward": reward,
                        'nothinking_reward': reward,
                        'thinking_reward': None,
                    })
                else:
                    score.update({
                        "reward": reward,
                        'nothinking_reward': None,
                        'thinking_reward': reward,
                    })
            else:
                reward = score["score"]
            # Store the information including original reward
            for key, value in score.items():
                reward_extra_info[key].append(value)

            reward_tensor[i, valid_response_length - 1] = reward
        # print("Label here: -------------------------------------------------------")
        # if self.is_training:
        #     print("Nothinking reward" + str(reward_extra_info['nothinking_reward']))
        #     print("Thinking reward" + str(reward_extra_info['thinking_reward']))
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

