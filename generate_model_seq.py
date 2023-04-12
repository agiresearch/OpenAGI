""" 
This code mainly used to do constraint generation for Flan-T5.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Wenyue Hua, Yingqiang Ge"
__copyright__ = "Copyright 2023, OpenAGI"
__date__ = "2023/04/12"
__license__ = "GPLv3"
__version__ = "0.0.1"

from typing import Dict, List
from types import MethodType
import torch
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



class SeqGen:
    def __init__(self, model, tokenizer, device):
        i2i_tasks = { 
            "input": "image",
            "output": "image",
            "task_list": [ 
                "Colorization,",  
                "Image Denoising,", 
                "Image Deblurring,",
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
                "Visual Question Answering,",
            ],
        }

        self.candidates = [
            i2i_tasks,
            i2t_tasks,
            t2t_tasks,
            t2i_tasks,
            # tt2t_tasks,
            # i2it_tasks,
            # it2i_tasks,
            # it2t_tasks,
        ]
        
        self.device = device
        self.model = model#.to(self.device)
        self.tokenizer = tokenizer
        
        
        
        
    def find_last_task(self,sentence):
        if sentence.count(6) == 1:
            last_cand = sentence[1 : sentence.index(6) + 1]
            return last_cand
        indices = [i for i, c in enumerate(sentence) if c == 6]
        last_cand = sentence[indices[-2] + 1 : indices[-1] + 1 :]
        return last_cand


    def t5_prefix_allowed_tokens_fn(self, module_length, constraint):
        all_candidates = [
            a for candidate_list in self.candidates for a in candidate_list["task_list"]
        ]

        def prefix_allowed_tokens(batch_id, sentence):
            sentence = sentence.tolist()
            # print(tokenizer.decode(sentence))
            if sentence.count(6) == 0:              
                all_candidate_trie = Trie(
                    [[0] + self.tokenizer.encode("{}".format(e)) for e in all_candidates]
                )
                trie_out = all_candidate_trie.get(sentence)
            elif sentence[-1] == 6 and sentence.count(6) != module_length:
                one_cand = self.find_last_task(sentence)
                one_cand = self.tokenizer.decode(one_cand)
                next_input_type = [
                    candidate_list
                    for candidate_list in self.candidates
                    if one_cand in candidate_list["task_list"]
                ][0]["output"]
                # find corresponding list
                one_candidate_list = [
                    candidate
                    for candidate_list in [
                        candidate_list["task_list"]
                        for candidate_list in self.candidates
                        if candidate_list["input"] == next_input_type
                    ]
                    for candidate in candidate_list
                ]
                    
                # remove candidates that occurred
                remove_repetition = [
                    candidate
                    for candidate in one_candidate_list
                    if candidate not in self.tokenizer.decode(sentence)
                ] 
                
                # print(remove_repetition)
                
                # if sentence.count(6) == 1:
                #     remove_repetition_ = remove_repetition[constraint[0]:constraint[1]] + ["</s>"]
                #     one_candidate_trie = Trie(
                #         [[0] + self.tokenizer.encode("{}".format(e)) for e in remove_repetition_]
                #     )
                # else:
                #     one_candidate_trie = Trie(
                #         [[0] + self.tokenizer.encode("{}".format(e)) for e in remove_repetition + ["</s>"]]
                #     )
                    
                one_candidate_trie = Trie([[0] + self.tokenizer.encode("{}".format(e)) for e in remove_repetition + ["</s>"]])
                trie_out = one_candidate_trie.get([0])
                
            elif sentence[-1] != 6 and sentence.count(6) != module_length:
                one_cand = self.find_last_task(sentence)
                one_cand = self.tokenizer.decode(one_cand)
                input_type = [
                    candidate_list
                    for candidate_list in self.candidates
                    if one_cand in candidate_list["task_list"]
                ][0]["output"]
                # find corresponding list
                one_candidate_list = [
                    candidate
                    for candidate_list in [
                        candidate_list["task_list"]
                        for candidate_list in self.candidates
                        if candidate_list["input"] == input_type
                    ]
                    for candidate in candidate_list
                ]
                # remove candidates that occurred
                remove_repetition = [
                    candidate
                    for candidate in one_candidate_list
                    if candidate not in self.tokenizer.decode(sentence)
                ]
                one_candidate_trie = Trie(
                    [[0] + self.tokenizer.encode("{}".format(e)) for e in remove_repetition]
                )
                indices = [i for i, c in enumerate(sentence) if c == 6]
                sentence = sentence[indices[-1] + 1 :]
                trie_out = one_candidate_trie.get([0] + sentence)
            elif sentence.count(6) == module_length:
                candidate_trie = Trie(
                    [[0] + self.tokenizer.encode("{}".format(e)) for e in ["</s>"]]
                )
                trie_out = candidate_trie.get([0])

            return trie_out

        return prefix_allowed_tokens


    def generate_sequence(self, input_s, \
                          module_length=5, \
                          beam_size=4, \
                          num_seq=1, \
                          top_k=5, \
                          top_p=0.9, \
                          temperature=0.7, \
                          constraint=[0,100], \
                          num_beam_groups=2):
        output_sequences = []
        log_probs = []
        # output_scores = []
        # output_results = []
        
        # for input_s in input_sentences:
        input_ids = self.tokenizer.batch_encode_plus(input_s, padding="longest", return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(self.device)
        prefix_allowed_tokens = self.t5_prefix_allowed_tokens_fn(module_length, constraint=constraint)
        output = self.model.generate_with_grad(
            input_ids,
            max_length=80,
            min_length=1,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=beam_size,
            num_return_sequences=num_seq,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            renormalize_logits=True,
            # do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stopping=True,
            # no_repeat_ngram_size=no_repeat_ngram_size, 
            num_beam_groups=num_beam_groups,
        )

        # print(output["sequences"])

        output_ids = output["sequences"][:,1:]
        output_result = [s for s in self.tokenizer.batch_decode(output_ids)]
        output_sequence = [s.replace("<pad>", "").replace("</s>", "") for s in self.tokenizer.batch_decode(output_ids)]
        output_sequences.append(output_sequence)

        # B * length tuple of (beam_size * vocab_size) tensor
        scores = output["scores"]
        if beam_size > 1:
            output_score = output.sequences_scores
            if num_seq == 1:
                length = output_ids.size(-1)
                logprob = 0
                # B * beam_size * length
                beam_indices = output["beam_indices"][0]
                for l in range(length):
                    beam_index = beam_indices[l]
                    score = scores[l][beam_index]
                    # score = toenrch.exp(scores[l][beam_index])  # unnormalized prob
                    # score /= score.sum()  # normalized prob
                    # score = torch.log(score)  # normalized log prob
                    if self.tokenizer.decode(output_ids[0][l]) == "</s>":
                        continue
                    logprob += score[output_ids[0][l]]
                    # else:
                        # logprob = 0
                loss = logprob#/length
                log_probs.append([loss])
            else:
                loss = []
                for i in range(num_seq):
                    if 0 in output_ids[i]:
                        one_length = output_ids[i][
                            : (output_ids[i] == 0).nonzero(as_tuple=True)[0].tolist()[0]
                        ].size(-1)
                    else:
                        one_length = output_ids[i].size(-1)
                    logprob = 0
                    # B * num_seq * length
                    beam_indices = output["beam_indices"][i]
                    for l in range(one_length):
                        beam_index = beam_indices[l]
                        score = scores[l][beam_index]
                        # score = torch.exp(scores[l][beam_index])  # unnormalized prob
                        # score /= score.sum()  # normalized prob
                        # score = torch.log(score)  # normalized log prob
                        if self.tokenizer.decode(output_ids[i][l]) == "</s>":
                            continue
                        logprob += score[output_ids[i][l]]
                    loss.append(logprob)#/one_length)

                log_probs.append(loss)
        else:
            logprob = 0
            length = output_ids.size(-1)
            for l in range(length):

                score = scores[l][0]
                if self.tokenizer.decode(output_ids[0][l]) == "</s>":
                    continue
                logprob += score[output_ids[0][l]]
            loss = logprob#/length
            log_probs.append([loss])
                
                    
        # return output_sequence, loss, output_score, output_result #, prob, output_score
        return output_sequence, loss
