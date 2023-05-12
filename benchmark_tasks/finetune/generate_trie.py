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

__authors__ = "Wenyue Hua, Yingqiang Ge"
__copyright__ = "Copyright 2023, OpenAGI"
__date__ = "2023/04/13"
__license__ = "Apache 2.0"
__version__ = "0.0.1"


from typing import Dict, List
from types import MethodType

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


"""
def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        if sentence[-1] == 6:  # last token is ","
            trie_out = candidate_trie.get([0])
        elif 6 not in sentence:  # "," is not in the generated sentence
            trie_out = candidate_trie.get(sentence)
        else:
            assert 6 in sentence
            indices = [i for i, c in enumerate(sentence) if c == 6]
            sentence = sentence[indices[-1] + 1 :]
            trie_out = candidate_trie.get([0] + sentence)
        return trie_out

    return prefix_allowed_tokens
"""


def find_last_task(sentence):
    if sentence.count(6) == 1:
        last_cand = sentence[1 : sentence.index(6) + 1]
        return last_cand
    indices = [i for i, c in enumerate(sentence) if c == 6]
    last_cand = sentence[indices[-2] + 1 : indices[-1] + 1 :]
    return last_cand


def t5_prefix_allowed_tokens_fn(candidates, tokenizer, module_length):
    all_candidates = [
        a for candidate_list in candidates for a in candidate_list["task_list"]
    ]

    def prefix_allowed_tokens(batch_id, sentence):
        # print('#####BATCH ID:'+str(batch_id))
        # print(len(all_candidates))
        
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
            trie_out = one_candidate_trie.get([0] + sentence)
        elif sentence.count(6) == module_length:
            candidate_trie = Trie(
                [[0] + tokenizer.encode("{}".format(e)) for e in ["</s>"]]
            )
            trie_out = candidate_trie.get([0])

        return trie_out

    return prefix_allowed_tokens


def bloom_prefix_allowed_tokens_fn(candidates, tokenizer):
    all_candidates = [
        a for candidate_list in candidates for a in candidate_list["task_list"]
    ]

    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        # print(tokenizer.decode(sentence))
        if sentence.count(6) == 0:
            all_candidate_trie = Trie(
                [tokenizer.encode("{}".format(e)) for e in all_candidates]
            )
            trie_out = all_candidate_trie.get(sentence)
        elif sentence[-1] == 6 and sentence.count(6) != 5:
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
            ]
            one_candidate_trie = Trie(
                [tokenizer.encode("{}".format(e)) for e in remove_repetition]
            )
            trie_out = one_candidate_trie.get([])
        elif sentence[-1] != 6 and sentence.count(6) != 5:
            one_cand = find_last_task(sentence)
            one_cand = tokenizer.decode(one_cand)
            input_type = [
                candidate_list
                for candidate_list in candidates
                if one_cand in candidate_list["task_list"]
            ][0]["input"]
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
                [tokenizer.encode("{}".format(e)) for e in remove_repetition]
            )
            indices = [i for i, c in enumerate(sentence) if c == 6]
            sentence = sentence[indices[-1] + 1 :]
            trie_out = one_candidate_trie.get([] + sentence)
        elif sentence.count(6) == 5:
            candidate_trie = Trie([tokenizer.encode("{}".format(e)) for e in "<eos>"])
            trie_out = candidate_trie.get([])

        return trie_out

    return prefix_allowed_tokens


"""
    candidates = [
        "image to caption",
        "image to caption, text to text",
        "image to caption, text to text, text to music",
        "text to text",
        "text to text, text to music",
        "text to text, text to music, music to image",
        "text to music, music to image",
    ]
"""

i2i_tasks = {
    "input": "image",
    "output": "image",
    "task_list": [
        "Colorization,",  
        "Image Denoising,", 
        "Image Delurring,",
        "Image Super Resolution,"
    ],
}

i2t_tasks = {
    "input": "image",
    "output": "text",
    "task_list": [
        "Image Classification,",  
        "Image Captioning,", 
        "Object Detection,"
    ],
}

t2t_tasks = {
    "input": "text",
    "output": "text",
    "task_list": [
        "Text Summarization,",  
        "Text Generation,", 
        "Machine Translation,",  
        "Fill Mask,",
        "Sentiment Analysis,"
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
    "task_list": [
        "Question Answering,", 
    ],
}


it2t_tasks = {
    "input": "image+text",
    "output": "text",
    "task_list": [
        "Visual Question Answering,",  #: [2, 1],
    ],
}

candidates = [
    i2i_tasks,
    i2t_tasks,
    t2t_tasks,
    t2i_tasks,
    # tt2t_tasks,
    # i2it_tasks,
    # it2i_tasks,
    # it2t_tasks,
]


def generate_sequence(
    input_ids, model, tokenizer, candidates, module_length=3, num_beams=4, num_return_sequences=2
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
    )
    
    output_sequences = []
    log_probs = []
    for i in range(num_return_sequences):
        output_ids = output["sequences"][i][1:]
        output_sequence = (
            tokenizer.decode(output_ids).replace("<pad>", "").replace("</s>", "")
        )
        # B * length * beam_size * vocab_size
        scores = output["scores"]
        # print(scores[0])
        length = output_ids.size(-1)
        logprob = 0
        if num_beams > 1:
            # B * beam_size * length
            beam_indices = output["beam_indices"][i]#[1:]
            # print(beam_indices)
            for l in range(length):
                beam_index = beam_indices[l]
                score = torch.exp(scores[l][beam_index])  # unnormalized prob
                if score.sum() != 0:
                    score /= score.sum()  # normalized prob
                    score = torch.log(score)  # normalized log prob
                    logprob += score[output_ids[l]]
                # else:
                #     logprob += 1
            loss = logprob
        else:
            for l in range(length):
                score = torch.exp(scores[l][0])  # unnormalized prob
                score /= score.sum()  # normalized prob
                score = torch.log(score)  # normalized log prob
                logprob += score[output_ids[l]]
        loss = logprob
        output_sequences.append(output_sequence)
        log_probs.append(loss)
    # return output_sequence, loss
    return output_sequences, log_probs


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    candidates = [
    i2i_tasks,
    i2t_tasks,
    t2t_tasks,
    t2i_tasks,
    # tt2t_tasks,
    # i2it_tasks,
    # it2i_tasks,
    # it2t_tasks,
]
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model = T5ForConditionalGeneration.from_pretrained("t5-large").cuda()
    # candidate_trie = Trie([[0] + tokenizer.encode("{}".format(e)) for e in candidates])
    # print(candidate_trie.trie_dict)

    generate_with_grad = undecorated(model.generate)
    model.generate_with_grad = MethodType(generate_with_grad, model)

    input_s = [
        "Given noisy image, how to return the regular image step by step?"
    ]
    input_ids = tokenizer.batch_encode_plus(
        input_s, padding="longest", return_tensors="pt"
    )["input_ids"].cuda()

    # module_length = 5
    # beam_size = 5
    output_sequence, loss = generate_sequence(input_ids, model, tokenizer, candidates)
    print(output_sequence)
    print(loss)
    
