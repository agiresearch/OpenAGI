"""
Copyright 2023 Yingqiang Ge

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

__author__ = "Wenyue Hua"
__copyright__ = "Copyright 2023, OpenAGI"
__date__ = "2023/05/13"
__license__ = "Apache 2.0"
__version__ = "0.0.1"



from typing import Dict, List
from types import MethodType
from torch.functional import _consecutive_return_output

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    BloomForCausalLM,
)
import torch
import time
from undecorated import undecorated


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)



def find_last_task(sentence):
    if sentence.count(6) == 1:
        last_cand = sentence[1 : sentence.index(6) + 1]
        if 61 in last_cand:
            last_cand.remove(61)
        if 41 in last_cand:
            last_cand.remove(41)
        return last_cand
    indices = [i for i, c in enumerate(sentence) if c == 6]
    last_cand = sentence[indices[-2] + 1 : indices[-1] + 1 :]
    if 61 in last_cand:
        last_cand.remove(61)
    if 41 in last_cand:
        last_cand.remove(41)
    return last_cand


def find_second_task(sentence):
    assert 61 in sentence
    end_position = sentence.index(61)
    start_positions = [1] + [i for i, c in enumerate(sentence[:end_position]) if c == 6]
    start_position = start_positions[-2]
    second_cand = sentence[start_position:end_position]
    if 61 in second_cand:
        second_cand.remove(61)
    if 41 in second_cand:
        second_cand.remove(41)
    if second_cand[0] == 6:
        second_cand = second_cand[1:]
    return second_cand


def pad_to_max_length(tokenizer, candidate_list):
    padd_candidate_list = []
    for one_candidate_list in candidate_list:
        candidates = one_candidate_list["task_list"]
        task_ids = tokenizer(candidates, padding="max_length", max_length=8)[
            "input_ids"
        ]
        padded_task_ids = []
        for i in range(len(task_ids)):
            if 0 in task_ids[i]:
                padded_number = task_ids[i].count(0)
                padded_task_ids.append(
                    task_ids[i][: task_ids[i].index(6)] + [2] * padded_number + [6]
                )
            else:
                padded_task_ids.append(task_ids[i][:-1])
        padded_task = tokenizer.batch_decode(padded_task_ids)
        one_candidate_list["task_list"] = padded_task
        padd_candidate_list.append(one_candidate_list)
    return padd_candidate_list


def check_two_input_types(sentence, tokenizer):
    first_cand = find_last_task(sentence)
    first_cand = tokenizer.decode(first_cand).strip()
    first_input_type = [
        candidate_list
        for candidate_list in candidates
        if first_cand in candidate_list["task_list"]
    ][0]["output"]
    second_cand = find_second_task(sentence)
    second_cand = tokenizer.decode(second_cand).strip()
    second_input_type = [
        candidate_list
        for candidate_list in candidates
        if second_cand in candidate_list["task_list"]
    ][0]["output"]
    # find corresponding list
    one_candidate_list = [
        candidate
        for candidate_list in [
            candidate_list["task_list"]
            for candidate_list in candidates
            if second_input_type + "+" + first_input_type == candidate_list["input"]
            or first_input_type + "+" + second_input_type == candidate_list["input"]
        ]
        for candidate in candidate_list
    ]
    # remove candidates that occurred
    remove_repetition = [
        candidate
        for candidate in one_candidate_list
        if candidate not in tokenizer.decode(sentence)
    ]
    one_candidate_trie = Trie(
        [[0] + tokenizer.encode("{}".format(e)) for e in remove_repetition]
    )
    indices = [i for i, c in enumerate(sentence) if c == 61]
    sentence = sentence[indices[-1] + 1 :]
    trie_out = one_candidate_trie.get([0] + sentence)
    return trie_out


def count_parallel_length(sentence):
    if sentence.count(41) == 2 and sentence.count(61) == 2:
        left_parenthesis_position = [i for i, c in enumerate(sentence) if c == 41]
        right_parenthesis_position = [i for i, c in enumerate(sentence) if c == 61]
        first_parallel_length = sentence[
            left_parenthesis_position[0] : right_parenthesis_position[0]
        ].count(6)
        second_parallel_length = sentence[
            left_parenthesis_position[1] : right_parenthesis_position[1]
        ].count(6)
        rest = sentence[right_parenthesis_position[1] :].count(6)
        if rest == 0:
            return 0
        return max(first_parallel_length, second_parallel_length) + rest
    elif sentence.count(61) == 0:
        return sentence.count(6)
    else:
        assert sentence.count(61) == 1
        second_parallel = sentence[sentence.index(61) + 1 :]
        return second_parallel.count(6)


def after_one_cand(sentence, tokenizer):
    one_cand = find_last_task(sentence)
    one_cand = tokenizer.decode(one_cand)
    input_type = [
        candidate_list
        for candidate_list in candidates
        if one_cand in candidate_list["task_list"]
    ][0]["output"]
    # find corresponding list
    one_candidate_list = [
        candidate
        for candidate_list in [
            candidate_list["task_list"]
            for candidate_list in candidates
            if candidate_list["input"] == input_type
        ]
        for candidate in candidate_list
    ]
    if sentence.count(61) == 0 or sentence.count(61) == 2:
        remove_repetition = [
            candidate
            for candidate in one_candidate_list
            if candidate not in tokenizer.decode(sentence)
        ]
    else:
        assert sentence.count(61) == 1
        sentence = sentence[sentence.index(61) + 1 :]
        remove_repetition = [
            candidate
            for candidate in one_candidate_list
            if candidate not in tokenizer.decode(sentence)
        ]

    return remove_repetition


def t5_prefix_allowed_tokens_fn(candidates, tokenizer, module_length):
    # candidates = pad_to_max_length(tokenizer, candidates)
    all_candidates = [
        a for candidate_list in candidates for a in candidate_list["task_list"]
    ]
    # pad all tasks to the same length

    def prefix_allowed_tokens(batch_id, sentence):
        # print(tokenizer.decode(sentence))
        # print("***")
        new_sentence_list = sentence.tolist()
        if len(sentence) <= 1:
            all_candidate_trie = Trie(
                [[0] + tokenizer.encode("({}".format(e)) for e in all_candidates]
                + [[0] + tokenizer.encode("({}".format(e)) for e in all_candidates]
            )
            trie_out = all_candidate_trie.get(new_sentence_list)

            return trie_out
        else:
            if sentence[1] == 41:
                return parenthesis_prefix_allowed_tokens(batch_id, sentence)
            else:
                return without_parenthesis_prefix_allowed_tokens(batch_id, sentence)

    def without_parenthesis_prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        # print(tokenizer.decode(sentence))
        if sentence.count(6) == 0:
            all_candidate_trie = Trie(
                [[0] + tokenizer.encode("{}".format(e)) for e in all_candidates]
            )
            trie_out = all_candidate_trie.get(sentence)
        elif sentence[-1] == 6 and sentence.count(6) != module_length:
            one_cand = find_last_task(sentence)
            one_cand = tokenizer.decode(one_cand)
            next_input_type = [
                candidate_list
                for candidate_list in candidates
                if one_cand in candidate_list["task_list"]
            ][0]["output"]
            # find corresponding list
            one_candidate_list = [
                candidate
                for candidate_list in [
                    candidate_list["task_list"]
                    for candidate_list in candidates
                    if candidate_list["input"] == next_input_type
                ]
                for candidate in candidate_list
            ]
            # remove candidates that occurred
            remove_repetition = [
                candidate
                for candidate in one_candidate_list
                if candidate not in tokenizer.decode(sentence)
            ] + ["</s>"]
            one_candidate_trie = Trie(
                [[0] + tokenizer.encode("{}".format(e)) for e in remove_repetition]
            )
            trie_out = one_candidate_trie.get([0])
        elif sentence[-1] != 6 and sentence.count(6) != module_length:
            # if sentence[-1] == 0 and sentence[sentence.index(0) - 1] != 6:
            #    sentence = sentence[: sentence.index(0)]
            # print(tokenizer.decode(sentence))
            one_cand = find_last_task(sentence)
            one_cand = tokenizer.decode(one_cand)
            input_type = [
                candidate_list
                for candidate_list in candidates
                if one_cand in candidate_list["task_list"]
            ][0]["output"]
            # find corresponding list
            one_candidate_list = [
                candidate
                for candidate_list in [
                    candidate_list["task_list"]
                    for candidate_list in candidates
                    if candidate_list["input"] == input_type
                ]
                for candidate in candidate_list
            ]
            # remove candidates that occurred
            remove_repetition = [
                candidate
                for candidate in one_candidate_list
                if candidate not in tokenizer.decode(sentence)
            ]
            one_candidate_trie = Trie(
                [[0] + tokenizer.encode("{}".format(e)) for e in remove_repetition]
            )
            indices = [i for i, c in enumerate(sentence) if c == 6]
            sentence = sentence[indices[-1] + 1 :]
            # a = one_candidate_trie.get([0] + sentence)
            # for b in a:
            #    print(tokenizer.decode(b))
            # print("***")
            trie_out = one_candidate_trie.get([0] + sentence)
        elif sentence.count(6) == module_length:
            candidate_trie = Trie(
                [[0] + tokenizer.encode("{}".format(e)) for e in ["</s>"]]
            )
            trie_out = candidate_trie.get([0])

        return trie_out

    def parenthesis_prefix_allowed_tokens(batch_id, sentence):
        # print(tokenizer.decode(sentence))
        # print("***")
        sentence = sentence.tolist()
        # either begin of sentence, or finish one ()
        if sentence.count(6) == 0 or (
            sentence[-1] == 61  # )
            and sentence.count(41) == 1  # has one ()
            and sentence.count(61) == 1
        ):
            all_candidate_trie = Trie(
                [[0] + tokenizer.encode("({}".format(e)) for e in all_candidates]
            )
            if 61 not in sentence:
                trie_out = all_candidate_trie.get(sentence)
            else:
                trie_out = all_candidate_trie.get([0])
        elif sentence[-1] != 6 and count_parallel_length(sentence) < module_length:
            if sentence[-1] == 61 and sentence.count(61) == 2:
                # check two input types and generate without any () in the future
                trie_out = check_two_input_types(sentence, tokenizer)
            else:
                # keep generating the unfinished task, can generate ) or not, not necessary
                if sentence.count(61) == 0:
                    remove_repetition = after_one_cand(sentence, tokenizer)
                    if sentence[-1] == 61:
                        one_candidate_trie = Trie(
                            [[0, 1]]
                            + [
                                [0] + tokenizer.encode("{}".format(e))
                                for e in remove_repetition
                            ]
                        )
                    else:
                        one_candidate_trie = Trie(
                            [
                                [0] + tokenizer.encode("{}".format(e))
                                for e in remove_repetition
                            ]
                        )
                    indices = [i for i, c in enumerate(sentence) if c == 6]
                    sentence = sentence[indices[-1] + 1 :]
                    trie_out = one_candidate_trie.get([0] + sentence)
                elif sentence.count(61) == 1:
                    # the first task of the second parallel
                    rebegin_sentence = sentence[sentence.index(61) + 1 :]
                    if 6 not in rebegin_sentence:
                        all_candidate_trie = Trie(
                            [
                                [0] + tokenizer.encode("({}".format(e))
                                for e in all_candidates
                            ]
                        )
                        sentence = sentence[sentence.index(61) + 1 :]
                        trie_out = all_candidate_trie.get([0] + rebegin_sentence)
                    else:
                        # the non-first task of the second parallel
                        remove_repetition = after_one_cand(sentence, tokenizer)
                        if sentence[-1] == 61 and sentence.count(61) == 1:
                            one_candidate_trie = Trie(
                                [[0, 1]]
                                + [
                                    [0] + tokenizer.encode("{}".format(e))
                                    for e in remove_repetition
                                ]
                            )
                        else:
                            one_candidate_trie = Trie(
                                [
                                    [0] + tokenizer.encode("{}".format(e))
                                    for e in remove_repetition
                                ]
                            )
                        indices = [i for i, c in enumerate(sentence) if c == 6]
                        sentence = sentence[indices[-1] + 1 :]
                        trie_out = one_candidate_trie.get([0] + sentence)
                else:
                    right_parentheses_position = [
                        index for index, c in enumerate(sentence) if c == 61
                    ]
                    if 6 not in sentence[right_parentheses_position[-1] :]:
                        # generate the first task after ()()
                        trie_out = check_two_input_types(sentence, tokenizer)
                    else:  # find the last task and generate without any () in the future
                        remove_repetition = after_one_cand(sentence, tokenizer)
                        one_candidate_trie = Trie(
                            [
                                [0] + tokenizer.encode("{}".format(e))
                                for e in remove_repetition
                            ]
                        )
                        indices = [i for i, c in enumerate(sentence) if c == 6]
                        sentence = sentence[indices[-1] + 1 :]
                        trie_out = one_candidate_trie.get([0] + sentence)
        elif sentence[-1] == 6 and count_parallel_length(sentence) < module_length:
            if sentence.count(41) - sentence.count(61) == 1:
                # need to generate candidates with ), without ), or directly )
                remove_repetition = after_one_cand(sentence, tokenizer)
                if count_parallel_length(sentence) + 1 >= module_length:
                    trie_out = [61]
                else:
                    one_candidate_trie = Trie(
                        [[0, 61]]
                        + [
                            [0] + tokenizer.encode("{}".format(e))
                            for e in remove_repetition
                        ]
                    )
                    indices = [i for i, c in enumerate(sentence) if c == 6]
                    sentence = sentence[indices[-1] + 1 :]
                    trie_out = one_candidate_trie.get([0] + sentence)
            else:
                # outside of (), find the last task and keep generating
                remove_repetition = after_one_cand(sentence, tokenizer)
                one_candidate_trie = Trie(
                    [[0] + tokenizer.encode("{}".format(e)) for e in remove_repetition]
                )
                indices = [i for i, c in enumerate(sentence) if c == 6]
                sentence = sentence[indices[-1] + 1 :]
                trie_out = one_candidate_trie.get([0] + sentence)
        else:
            assert count_parallel_length(sentence) >= module_length
            if sentence.count(41) - sentence.count(61) == 1:
                trie_out = [61]
            else:
                trie_out = [1]

        return trie_out

    return prefix_allowed_tokens


i2i_tasks = {
    "input": "image",
    "output": "image",
    "task_list": [
        "Colorization,",
        "Image Denoising,",
        "Image Delurring,",
        "Image Super Resolution,",
    ],
}

i2t_tasks = {
    "input": "image",
    "output": "text",
    "task_list": ["Image Classification,", "Image Captioning,", "Object Detection,"],
}

t2t_tasks = {
    "input": "text",
    "output": "text",
    "task_list": [
        "Text Summarization,",
        "Text Generation,",
        "Machine Translation,",
        "Fill Mask,",
        "Sentiment Analysis,",
    ],
}

t2i_tasks = {
    "input": "text",
    "output": "image",
    "task_list": ["Text to Image Generation,"],
}

tt2t_tasks = {
    "input": "text+text",
    "output": "text",
    "task_list": ["Question Answering,",],
}


it2t_tasks = {
    "input": "image+text",
    "output": "text",
    "task_list": ["Visual Question Answering,",],
}

candidates = [
    i2i_tasks,
    i2t_tasks,
    t2t_tasks,
    t2i_tasks,
    tt2t_tasks,
    # i2it_tasks,
    # it2i_tasks,
    it2t_tasks,
]


def generate_sequence_transition_scores(
    input_ids,
    model,
    tokenizer,
    candidates,
    module_length,
    num_beams,
    num_return_sequences,
):
    prefix_allowed_tokens = t5_prefix_allowed_tokens_fn(
        candidates, tokenizer, module_length
    )
    output = model.generate_with_grad(
        input_ids,
        max_length=80,
        min_length=1,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
        renormalize_logits=True,
    )

    output_ids = output["sequences"][:, 1:]
    output_sequence = [
        s.replace("<pad>", "").replace("</s>", "").replace("<unk>", "")
        for s in tokenizer.batch_decode(output_ids)
    ]
    print(output_sequence)

    if num_beams > 1:
        transition_scores = model.compute_transition_scores(
            sequences=output.sequences,
            scores=output.scores,
            beam_indices=output.beam_indices,
            normalize_logits=True,
        )
    else:
        transition_scores = model.compute_transition_scores(
            sequences=output.sequences, scores=output.scores, normalize_logits=True,
        )
    remove_nan_inf = torch.nan_to_num(transition_scores)
    logprob = torch.sum(remove_nan_inf, dim=-1)
    loss = -1 * logprob.sum()

    return output_sequence, loss


def generate_sequence_renormalize_logits(
    input_ids,
    model,
    tokenizer,
    candidates,
    module_length,
    num_beams,
    num_return_sequences,
):
    prefix_allowed_tokens = t5_prefix_allowed_tokens_fn(
        candidates, tokenizer, module_length
    )
    output = model.generate_with_grad(
        input_ids,
        max_length=80,
        min_length=2,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
        renormalize_logits=True,
    )
    output_ids = output["sequences"][:, 1:]
    output_sequence = [
        s.replace("<pad>", "").replace("</s>", "")
        for s in tokenizer.batch_decode(output_ids)
    ]
    # B * length tuple of (num_beams * vocab_size) tensor
    scores = output["scores"]
    if num_beams > 1:
        if num_return_sequences == 1:
            length = output_ids.size(-1)
            number_of_output_ids = len(output_ids[0].tolist())  # .count(6)
            logprob = 0
            # B * num_beams * length
            beam_indices = output["beam_indices"][0]
            for l in range(length):
                beam_index = beam_indices[l]
                if tokenizer.decode(output_ids[0][l]) in ["</s>", "<pad>"]:
                    continue
                else:
                    logprob += scores[l][beam_index][output_ids[0][l]]
            loss = logprob / number_of_output_ids
        else:
            loss = []
            for i in range(num_return_sequences):
                if 0 in output_ids[i]:
                    one_length = output_ids[i][
                        : (output_ids[i] == 0).nonzero(as_tuple=True)[0].tolist()[0]
                    ].size(
                        -1
                    )  # remove pad
                else:
                    one_length = output_ids[i].size(-1)
                number_of_output_ids = len(output_ids[i].tolist())  # .count(6)
                logprob = 0
                # B * num_return_sequences * length
                beam_indices = output["beam_indices"][i]
                for l in range(one_length):
                    beam_index = beam_indices[l]
                    if tokenizer.decode(output_ids[i][l]) in ["</s>", "<pad>"]:
                        continue
                    else:
                        logprob += scores[l][beam_index][output_ids[i][l]]
                loss.append(logprob / number_of_output_ids)
    else:
        logprob = 0
        number_of_output_ids = len(output_ids[0].tolist())  # .count(6)
        length = output_ids.size(-1)
        for l in range(length):
            if tokenizer.decode(output_ids[0][l]) in ["</s>", "<pad>"]:
                continue
            else:
                logprob += scores[l][0][output_ids[0][l]]
        loss = logprob / number_of_output_ids

    return output_sequence, loss


def generate_sequence(
    input_ids,
    model,
    tokenizer,
    candidates,
    module_length,
    num_beams,
    num_return_sequences,
):
    prefix_allowed_tokens = t5_prefix_allowed_tokens_fn(
        candidates, tokenizer, module_length
    )
    output = model.generate_with_grad(
        input_ids,
        max_length=80,
        min_length=2,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
    )
    output_ids = output["sequences"][:, 1:]
    output_sequence = [
        s.replace("<pad>", "").replace("</s>", "")
        for s in tokenizer.batch_decode(output_ids)
    ]
    # B * length tuple of (num_beams * vocab_size) tensor
    scores = output["scores"]
    if num_beams > 1:
        if num_return_sequences == 1:
            length = output_ids.size(-1)
            number_of_output_ids = output_ids[0].tolist().count(6)
            logprob = 0
            # B * num_beams * length
            beam_indices = output["beam_indices"][0]
            for l in range(length):
                beam_index = beam_indices[l]
                # print(tokenizer.decode(output_ids[0][l]))
                # print(scores[l][beam_index][output_ids[0][l]])
                exponential_score = (
                    torch.exp(scores[l][beam_index]) + 1e-10
                )  # unnormalized prob
                # print(exponential_score[output_ids[0][l]])
                normalized_score = (
                    exponential_score / exponential_score.sum()
                )  # normalized prob
                # print(normalized_score[output_ids[0][l]])
                prob_score = torch.log(normalized_score)  # normalized log prob
                if tokenizer.decode(output_ids[0][l]) == "</s>":
                    continue
                else:
                    logprob += prob_score[output_ids[0][l]]
            loss = logprob / number_of_output_ids
        else:
            loss = []
            for i in range(num_return_sequences):
                if 0 in output_ids[i]:
                    one_length = output_ids[i][
                        : (output_ids[i] == 0).nonzero(as_tuple=True)[0].tolist()[0]
                    ].size(-1)
                else:
                    one_length = output_ids[i].size(-1)
                number_of_output_ids = output_ids[i].tolist().count(6)
                if number_of_output_ids == 0:
                    number_of_output_ids += 1
                logprob = 0
                # B * num_return_sequences * length
                beam_indices = output["beam_indices"][i]
                for l in range(one_length):
                    # print(tokenizer.decode(output_ids[i][l]))
                    beam_index = beam_indices[l]
                    # print(scores[l][beam_index])
                    exponential_score = (
                        torch.exp(scores[l][beam_index]) + 1e-10
                    )  # unnormalized prob
                    # print(exponential_score)
                    normalized_score = (
                        exponential_score / exponential_score.sum()
                    )  # normalized prob
                    # print(normalized_score)
                    prob_score = torch.log(exponential_score)  # normalized log prob
                    # print(prob_score)
                    if tokenizer.decode(output_ids[i][l]) == "</s>":
                        continue
                    else:
                        logprob += prob_score[output_ids[i][l]]
                loss.append(logprob / number_of_output_ids)
    else:
        logprob = 0
        number_of_output_ids = output_ids[0].tolist().count(6)
        length = output_ids.size(-1)
        for l in range(length):
            exponential_score = torch.exp(scores[l][0]) + 1e-10  # unnormalized prob
            normalized_score = exponential_score / (
                exponential_score.sum()
            )  # normalized prob
            prob_score = torch.log(normalized_score)  # normalized log prob
            if tokenizer.decode(output_ids[0][l]) == "</s>":
                continue
            else:
                logprob += prob_score[output_ids[0][l]]
        loss = logprob / number_of_output_ids

    return output_sequence, loss


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    # model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

    candidates = [
        i2i_tasks,
        i2t_tasks,
        t2t_tasks,
        t2i_tasks,
        tt2t_tasks,
        # i2it_tasks,
        # it2i_tasks,
        it2t_tasks,
    ]

    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model = T5ForConditionalGeneration.from_pretrained("t5-large").cuda()
    # model.load_state_dict(torch.load("15_shot_finetuned.pt"))
    # candidate_trie = Trie([[0] + tokenizer.encode("{}".format(e)) for e in candidates])
    # print(candidate_trie.trie_dict)

    generate_with_grad = undecorated(model.generate)
    model.generate_with_grad = MethodType(generate_with_grad, model)

    input_s = [
        # "Given blurry black&white image, how to return the inpainting of clear colored image in multiple steps? Image Deblurring, Colorization, Image Inpainting. \n Given blurry black&white image, how to return the outpainting of clear colored image in multiple steps? Image Deblurring, Colorization, Image Outpainting. \n Given blurry black&white image, how to return completed Chinese text in multiple steps? Image Deblurring, Colorization, Image Caption, Translation. \n Given blurry black&white image, how to return completed English text in multple steps? Image Deblurring, Colorization, Image Caption. \n Given clear black&white image, how to return the inpainting of clear colored image in multiple steps? Colorization, Image Inpainting. \n Given clear black&white image, how to return the outpainting of clear colored image in multiple steps? Colorization, Image Outpainting. \n Given blurry black&white image, how to return the painting of clear colored image in multiple steps? ",
        "Given blurry black&white image, how to return the inpainting of clear colored image in multiple steps?"
    ]
    input_ids = tokenizer.batch_encode_plus(
        input_s, padding="longest", return_tensors="pt"
    )["input_ids"].cuda()
    print(tokenizer.batch_decode(input_ids))

    module_length = 10
    num_beams = 2
    num_return_sequences = 2

    prefix_allowed_tokens = t5_prefix_allowed_tokens_fn(
        candidates, tokenizer, module_length
    )

    output_sequence, loss = generate_sequence(
        input_ids,
        model,
        tokenizer,
        candidates,
        module_length,
        num_beams,
        num_return_sequences,
    )

    """
    output = model.generate(
        input_ids,
        max_length=100,
        min_length=1,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        num_beams=num_beams,
        num_return_sequences=1,
        # return_dict_in_generate=True,
        # output_scores=True,
        # output_hidden_states=True,
    )
    """

    print(output_sequence)
