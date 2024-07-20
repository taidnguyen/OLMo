import abc
import logging
import re
import os, json, pathlib, requests, zipfile, io

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Union

import datasets
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchmetrics import Metric

from olmo.util import load_hf_dataset

from ..tokenizer import Tokenizer

from .hendrycks_math_util import (
    HENDRYCKS_STOP_SEQS, is_equiv, remove_boxed, last_boxed_only_string, strip_string
)

log = logging.getLogger(__name__)


class ICLMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False

    def __init__(self, tokenizer: Tokenizer=None, metric_type="acc") -> None:
        """
        Tai adds: exact_match_hendrycks_math
        metric_type: f1, acc, len_norm, pmi_dc, ce_loss"""
        super().__init__(sync_on_compute=True)

        self.metric_type = metric_type
        self.tokenizer = tokenizer

        self.add_state("loglikelihoods", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("exact_match_scores", default=[], dist_reduce_fx=None)

    def reset(self):
        self.loglikelihoods = []
        self.labels = []
        self.exact_match_scores = []

    def update(self, batch: Dict[str, Any], lm_logits: torch.Tensor, dc_lm_logits=None):
        if "exact_match" in self.metric_type:
            gen_ids = batch["generation_ids"].tolist()
            for idx, gen_id in enumerate(gen_ids):
                # Flatten the list if necessary
                if isinstance(gen_id[0], list):
                    gen_id = [token for sublist in gen_id for token in sublist]
                pred_answer = self.tokenizer.decode(gen_id, skip_special_tokens=True)
                label_answer = self.tokenizer.decode(batch["label_gen_ids"][idx], skip_special_tokens=True)

                if "hendrycks_math" in self.metric_type:
                    try:
                        pred_answer = remove_boxed(last_boxed_only_string(pred_answer))
                        label_answer = remove_boxed(last_boxed_only_string(label_answer))
                    except:
                        pass
                    # print(f"pred_answer: {pred_answer}, label_answer: {label_answer}")
                    exact_match_score = 1 if is_equiv(pred_answer, label_answer) else 0
                else:
                    exact_match_score = 1 if pred_answer == label_answer else 0
                self.exact_match_scores.append(exact_match_score)
        else:
            lm_logits = F.log_softmax(lm_logits, dim=-1)

            if self.metric_type == "pmi_dc":
                assert dc_lm_logits is not None, "PMI_DC acc type selected but no domain conditional logits provided"

            for idx, (doc_id, cont_id) in enumerate(zip(batch["doc_id"], batch["cont_id"])):
                # [cont_len]: continuation is padded for batching
                cont_tokens = batch["continuation"][idx][: batch["cont_len"][idx]]
                # get logits from LM for the continuation: [cont_len, vocab]
                # batch['input_ids'][idx] -> ctx + cont + padding
                # -1 in both indices: lm_logits will be left shited 1 pos as 0th pos in input generates next token in the 0th pos of lm_logits
                lm_cont_logits = lm_logits[idx][
                    batch["ctx_len"][idx] - 1 : batch["ctx_len"][idx] + batch["cont_len"][idx] - 1
                ]

                log_likelihood: torch.Tensor
                if self.metric_type == "pmi_dc":
                    assert dc_lm_logits is not None
                    # get domain conditional continuation logits: [cont_len, vocab]
                    dc_lm_cont_logits = dc_lm_logits[idx][
                        batch["dc_len"][idx] - 1 : batch["dc_len"][idx] + batch["cont_len"][idx] - 1
                    ]

                    # gather log-probs at continuation token indices but divide by domain conditional prob
                    log_likelihood = (
                        torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                        / torch.gather(dc_lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                    )
                elif self.metric_type == "acc" or self.metric_type == "f1":
                    # gather log-probs at continuation token indices
                    log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
                elif self.metric_type == "len_norm" or self.metric_type == "ce_loss":
                    log_likelihood = (
                        torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum() / batch["cont_str_len"][idx]
                    )
                    if self.metric_type == "ce_loss":
                        log_likelihood = -log_likelihood
                else:
                    raise ValueError(self.metric_type)

                # because metric states cannot be dict/list of tuples, store this tuple as tensor: (doc_id, cont_id, metric_state)
                self.loglikelihoods.append(
                    torch.Tensor((doc_id, cont_id, log_likelihood)).to(batch["continuation"][idx].device)
                )
                self.labels.append(
                    torch.LongTensor((doc_id, cont_id, batch["label_id"][idx])).to(batch["label_id"][idx].device)
                )

    def compute(self) -> torch.Tensor:
        if "exact_match" in self.metric_type:
            return torch.tensor(sum(self.exact_match_scores) / len(self.exact_match_scores))

        # states should have been synced from all accelerators at this point
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        loglikelihood_dict: Dict[int, Dict[int, float]] = {}
        label_dict = {}

        # collect labels
        for doc_id, cont_id, label_id in self.labels:
            if doc_id.item() not in label_dict:
                label_dict[doc_id.item()] = label_id.item()

        # collect loglikelihoods
        for doc_id, cont_id, loglikelihood in self.loglikelihoods:
            if int(doc_id.item()) not in loglikelihood_dict:
                loglikelihood_dict[int(doc_id.item())] = {}

            if int(cont_id.item()) not in loglikelihood_dict[int(doc_id.item())]:
                loglikelihood_dict[int(doc_id.item())][int(cont_id.item())] = loglikelihood

        # compute acc
        correct = []
        preds: Optional[List[float]] = None
        labels: Optional[List[int]] = None
        if self.metric_type == "f1":
            preds = []
            labels = []

        for doc_id in loglikelihood_dict:
            # each doc_id might have a different number of continuation
            num_continuations = len(loglikelihood_dict[doc_id].keys())
            loglikelihoods = torch.tensor([-float("inf")] * num_continuations)

            skip_document = False
            for cont_id in loglikelihood_dict[doc_id]:
                try:
                    loglikelihoods[cont_id] = loglikelihood_dict[doc_id][cont_id]
                except IndexError:
                    # We didn't process all of the continuations, so skip this document.
                    skip_document = True
                    break

            if skip_document:
                continue
            if self.metric_type == "ce_loss":
                correct.append(loglikelihoods[0])  # Only one answer is scored
            else:
                correct.append(1.0 if torch.argmax(loglikelihoods).item() == label_dict[doc_id] else 0.0)

            if self.metric_type == "f1":
                assert preds is not None
                assert labels is not None
                preds.append(torch.argmax(loglikelihoods).item())
                labels.append(label_dict[doc_id])

        if self.metric_type == "f1":
            assert preds is not None
            assert labels is not None
            # for NLI tasks, continuations are yes, no, neither, so idx=0 assigned to pos label
            score = f1_score(labels, preds, pos_label=0)
        else:
            score = sum(correct) / len(correct)

        return torch.tensor(score)


class ICLMultiChoiceTaskDataset(metaclass=abc.ABCMeta):
    """Only supports zero-shot for now."""

    metric_type: ClassVar[str]

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split="validation",
        prompts=[None],  # List of prompt variants to use
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.prompts = prompts
        self.current_prompt = None
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        dataset_list = []
        for ds_name in dataset_names:
            dataset = load_hf_dataset(self.dataset_path, ds_name, split)
            dataset_list.append(dataset)
        self.dataset = datasets.concatenate_datasets(dataset_list)

        # prep examples
        self.prep_examples()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        doc_id = 0
        for doc in self.dataset:
            for prompt in self.prompts:
                self.current_prompt = prompt
                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                continuations = self.doc_to_continuations(doc)
                label_id = self.doc_to_label(doc)
                doc_text = self.doc_to_text(doc)
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                        + f"\ndoc_text: {doc_text}\ncontinuations: {continuations}"
                    )

                for cont_id, continuation_str in enumerate(continuations):
                    cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                    continuation = self.token_encode(continuation_str)

                    # query, remove last token from continuation, truncate from left is longer than model ctx length
                    query = ctx + continuation[:-1]
                    query = query[-self.model_ctx_len :]
                    # this will be different from len(ctx) when truncated by model_ctx_len
                    actual_ctx_len = len(query) - len(continuation) + 1

                    # get domain conditional query
                    # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                    dc_query = dc + continuation[:-1]

                    # form a sample
                    self.samples.append(
                        {
                            "doc_id": doc_id,
                            "cont_id": cont_id,
                            "ctx": ctx,
                            "continuation": continuation,
                            "ctx_len": actual_ctx_len,
                            "dc_len": len(dc),
                            "cont_len": len(
                                continuation
                            ),  # even if query has last token removed, LM will output same cont len
                            "cont_str_len": cont_str_len,
                            "query": query,  # remove last token from continuation
                            "dc_query": dc_query,
                            "label_id": label_id,
                        }
                    )

                doc_id += 1

    def pad_tokens_until_max(self, tokens, max_len=2048):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        queries are already truncated at max length of model_ctx_len
        this acts as additional check for all types of sequences in the batch
        """
        if len(tokens) > self.model_ctx_len:
            return tokens[-self.model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded self.model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > self.model_ctx_len
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[: self.model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        # 'ctx', 'continuation', 'query' can all have variable length
        max_ctx_len = 0
        max_cont_len = 0
        max_query_len = 0
        max_dc_query_len = 0

        for sample in data:
            if len(sample["ctx"]) > max_ctx_len:
                max_ctx_len = len(sample["ctx"])

            if len(sample["continuation"]) > max_cont_len:
                max_cont_len = len(sample["continuation"])

            if len(sample["query"]) > max_query_len:
                max_query_len = len(sample["query"])

            if len(sample["dc_query"]) > max_dc_query_len:
                max_dc_query_len = len(sample["dc_query"])

        doc_ids = []
        cont_ids = []
        ctxs = []
        continuations = []
        ctx_lens = []
        dc_lens = []
        cont_lens = []
        cont_str_lens = []
        queries = []
        dc_queries = []
        label_ids = []

        # pad according to max_lengths
        for sample in data:
            doc_ids.append(sample["doc_id"])
            cont_ids.append(sample["cont_id"])

            ctxs.append(torch.LongTensor(self.pad_tokens_until_max(sample["ctx"], max_len=max_ctx_len)))
            continuations.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["continuation"], max_len=max_cont_len))
            )

            ctx_lens.append(sample["ctx_len"])
            dc_lens.append(sample["dc_len"])
            cont_lens.append(sample["cont_len"])
            cont_str_lens.append(sample["cont_str_len"])

            queries.append(torch.LongTensor(self.pad_tokens_until_max(sample["query"], max_len=max_query_len)))
            dc_queries.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["dc_query"], max_len=max_dc_query_len))
            )

            label_ids.append(sample["label_id"])

        batch = {
            "doc_id": torch.LongTensor(doc_ids),
            "cont_id": torch.LongTensor(cont_ids),
            "ctx": torch.stack(ctxs),
            "continuation": torch.stack(continuations),
            "ctx_len": torch.LongTensor(ctx_lens),
            "dc_len": torch.LongTensor(dc_lens),
            "cont_len": torch.LongTensor(cont_lens),  # since query has last token removed from continuation
            "cont_str_len": torch.LongTensor(cont_str_lens),
            "input_ids": torch.stack(queries),
            "dc_input_ids": torch.stack(dc_queries),
            "label_id": torch.LongTensor(label_ids),
        }

        return batch

    def token_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @abc.abstractmethod
    def doc_to_text(self, doc) -> str:
        """Match EAI eval harness
        returns a single context string
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_continuations(self, doc) -> List[str]:
        """Match EAI eval harness
        returns a list of continuations
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_label(self, doc) -> int:
        """Match EAI eval harness
        returns continuation id which corresponds to true label
        """
        raise NotImplementedError

    def doc_to_domain_conditional(self, doc) -> str:
        """Provide string for domain conditional normalization
        by default its blank string, continuation normalized by prob conditioned on a blank
        """
        del doc
        return " "


class PIQA(ICLMultiChoiceTaskDataset):
    """PIQA sends context in the following fashion: "Question: GOAL\nAnswer:"
    space added as prefix to each continuation

    implement PMI_DC

    {
        'goal': "How do I ready a guinea pig cage for it's new occupants?",
        'sol1': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.',
        'sol2': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.',
        'label': 0
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="piqa",
        dataset_name="plain_text",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + doc["sol1"], " " + doc["sol2"]]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class HellaSwag(ICLMultiChoiceTaskDataset):
    """HellaSwag concats "ACTIVITY_LABEL: CTX_A CTX_B.capitalize()" to form context and then sends endings as continuations
        space added as prefix to each continuation

    {
        'activity_label': 'Roof shingle removal',
        'ctx_a': 'A man is sitting on a roof.',
        'ctx_b': 'he',
        'ctx': 'A man is sitting on a roof. he',
        'endings': ['is using wrap to wrap a pair of skis.', 'is ripping level tiles off.', "is holding a rubik's cube.", 'starts pulling up roofing on a roof.'],
        'label': '3'
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="hellaswag",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")

        return text

    def doc_to_text(self, doc):
        return self.preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + self.preprocess(ending) for ending in doc["endings"]]

    def doc_to_label(self, doc):
        return int(doc["label"])

    def doc_to_domain_conditional(self, doc):
        domain_conditional = self.preprocess(doc["ctx_b"].capitalize())

        # ensure non 0 len domain conditional
        if len(domain_conditional) == 0:
            return self.preprocess(doc["ctx_a"]).split(" ")[-1]

        return domain_conditional


class WinoGrande(ICLMultiChoiceTaskDataset):
    """Prompt: split sentence at _ "SENTENCE[:idx] + OPTION1/OPTION2", where idx = SENTENCE.index("_")
        implement PMI_DC
        acc, random at 50%
        continuation is everything in setnence after '_' (" SENTENCE[idx:].strip()")

        Req_loglikelihood('People think Samantha', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')
        Req_loglikelihood('People think Rebecca', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')

    {
        'sentence': 'People think _ is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.',
        'option1': 'Samantha',
        'option2': 'Rebecca',
        'answer': '2'
    }

    TODO: might need to write custom metric for Winogrande
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="winogrande",
        dataset_name="winogrande_xl",
    ):
        # all winogrande datasets have same val set
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def prep_examples(self):
        """Overwrite for WinoGrande as multiple ctx, single continuation"""
        doc_id = 0
        for doc in self.dataset:
            # here ctx is a list
            ctxs = self.doc_to_text(doc)
            dcs = self.doc_to_domain_conditional(doc)

            continuation = self.doc_to_continuations(doc)
            label_id = self.doc_to_label(doc)
            cont_str_len = len(continuation) - 1  # continuations contain leading blank space

            # tokenize
            continuation = self.token_encode(continuation)

            for cont_id, (ctx, dc) in enumerate(zip(ctxs, dcs)):
                ctx = self.token_encode(ctx)
                dc = self.token_encode(dc)

                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation[:-1]
                query = query[-self.model_ctx_len :]

                # get domain conditional query
                # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                dc_query = dc + continuation[:-1]

                # form a sample
                self.samples.append(
                    {
                        "doc_id": doc_id,
                        "cont_id": cont_id,
                        "ctx": ctx,
                        "continuation": continuation,
                        "ctx_len": len(ctx),
                        "dc_len": len(dc),
                        "cont_len": len(
                            continuation
                        ),  # even if query has last token removed, LM will output same cont len
                        "cont_str_len": cont_str_len,
                        "query": query,  # remove last token from continuation
                        "dc_query": dc_query,
                        "label_id": label_id,
                    }
                )

            doc_id += 1

    def doc_to_text(self, doc):
        # special case where there are multiple ctx and single continuation
        pronoun_loc = doc["sentence"].index("_")

        ctx = []
        for option in [doc["option1"], doc["option2"]]:
            ctx.append(doc["sentence"][:pronoun_loc] + option)

        return ctx

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        pronoun_loc = doc["sentence"].index("_") + 1
        return " " + doc["sentence"][pronoun_loc:].strip()

    def doc_to_label(self, doc):
        return int(doc["answer"]) - 1

    def doc_to_domain_conditional(self, doc):
        """same number of domain conditionals as context"""
        return [doc["option1"], doc["option2"]]


class OpenBookQA(ICLMultiChoiceTaskDataset):
    """OBQA: question_stem is sent as context (no special prompt format) and choices are sent as continuation
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question_stem': 'Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as',
        'choices': {'text': ['Deep sea animals', 'fish', 'Long Sea Fish', 'Far Sea Animals'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="openbookqa",
        dataset_name="main",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["question_stem"]

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + choice for choice in doc["choices"]["text"]]

    def doc_to_label(self, doc):
        return ["A", "B", "C", "D"].index(doc["answerKey"].strip())

    def doc_to_domain_conditional(self, doc):
        return doc["question_stem"].strip().split(" ")[-1]


class BoolQ(ICLMultiChoiceTaskDataset):
    """Prompt: "PASSAGE\nQuestion: QUESTION?\nAnswer:"
    acc, random at 50% (SuperGLUE)
    continuation: yes, no

    {
        'question': 'is ncis new orleans over for the season',
        'passage': 'NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.',
        'label': 1
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="boolq",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["passage"] + "\nQuestion: " + doc["question"] + "?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['answer'] is True, return index of " yes" which is 0
        if doc["answer"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class SciQ(ICLMultiChoiceTaskDataset):
    """SciQ sends context as "SUPPORT\nQuestion: QUESTION\nAnswer:" and then distractors + correct_answer as continuations
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question': 'Who proposed the theory of evolution by natural selection?',
        'distractor3': 'Scopes',
        'distractor1': 'Linnaeus',
        'distractor2': 'shaw',
        'correct_answer': 'darwin',
        'support': ''
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="sciq",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["support"].strip() + "\nQuestion: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [
            " " + doc["distractor1"],
            " " + doc["distractor2"],
            " " + doc["distractor3"],
            " " + doc["correct_answer"],
        ]

    def doc_to_label(self, doc):
        del doc
        return 3

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class ArcEasy(ICLMultiChoiceTaskDataset):
    """ArcEasy creates context with "Question: QUESTION\nAnswer:" and sends the choices as continuations
        space added as prefix to each continuation

    {
        'question': 'Which technology was developed most recently?',
        'choices': {'text': ['cellular telephone', 'television', 'refrigerator', 'airplane'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="ai2_arc",
        dataset_name="ARC-Easy",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + choice for choice in doc["choices"]["text"]]

    def doc_to_label(self, doc):
        # some doc["answerKey"] are stored as numbers
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

        if doc["answerKey"] in num_to_letter:
            doc["answerKey"] = num_to_letter[doc["answerKey"]]

        return ["A", "B", "C", "D", "E"].index(doc["answerKey"])

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class ArcChallenge(ArcEasy):
    """ArcChallenge follows the same prompt format as ArcEasy.
    implement PMI_DC
    """

    metric_type = "len_norm"  # Ideally "pmi_dc"

    def __init__(
        self,
        tokenizer,
        dataset_path="ai2_arc",
        dataset_name="ARC-Challenge",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )


class ArcEasyCELoss(ArcEasy):
    """ArcEasyCELoss is ARCEasy using an alternate ce_loss metric"""

    metric_type = "ce_loss"

    def doc_to_continuations(self, doc):
        # We only consider the correct answer for this metric
        answer = doc["choices"]["text"][self.doc_to_label(doc)]
        return [" " + answer]

    def doc_to_label(self, doc):
        return 0


class BasicArithmetic(ArcEasy):
    """This is a basic arithmetic task follows the same prompt format as ArcEasy.
    Example:
    {"id": "q85_1d1d_max1d_plus",
    "question": "Calculate 2 + 5 =",
    "choices": {"text": ["8", "7", "6", "17"],
    "label": ["A", "B", "C", "D"]},
    "answerKey": "B", "type_tag": "easy"}

    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="allenai/basic_arithmetic",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )


class CommonsenseQA(ArcEasy):
    """CommonsenseQA
    Example:
    {'id': 'e68fb2448fd74e402aae9982aa76e527',
    'question': 'Where are  you likely to find a hamburger?',
    'question_concept': 'hamburger',
    'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
    'text': ['fast food restaurant', 'pizza', 'ground up dead cows', 'mouth', 'cow carcus']},
    'answerKey': 'A'}
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="tau/commonsense_qa",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )


class SocialIQa(ICLMultiChoiceTaskDataset):
    """SocialIQa
    Example:
    {'context': 'Jordan was in charge of taking the food on the camping trip and left all the food at home.',
     'question': 'How would Jordan feel afterwards?',
     'answerA': 'horrible that he let his friends down on the camping trip',
     'answerB': "happy that he doesn't need to do the cooking on the trip",
     'answerC': 'very proud and accomplished about the camping trip', 'label': '1'}
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="social_i_qa",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "Question: " + doc["context"] + " " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [
            " " + doc["answerA"],
            " " + doc["answerB"],
            " " + doc["answerC"],
        ]

    def doc_to_label(self, doc):
        return int(doc["label"]) - 1

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


class COPA(ICLMultiChoiceTaskDataset):
    """Prompt: "PREMISE.strip()[:-1] because/therefore"
    Req_loglikelihood('The pair of students came under scrutiny by the teacher because', ' the students both received excellent grades.'
    continuations: CHOICE1/CHOICE2

    "cause": "because",
    "effect": "therefore",

    implement PMI_DC
    acc, random at 50%

    {
        'premise': 'The pair of students came under scrutiny by the teacher.',
        'choice1': 'The students both received excellent grades.',
        'choice2': 'Their responses on the assignment were identical.',
        'question': 'cause',
        'label': 1
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="copa",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        connector = "because" if doc["question"] == "cause" else "therefore"

        # remove the period
        return doc["premise"].strip()[:-1] + " " + connector

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        def convert_choice(choice):
            return choice[0].lower() + choice[1:]

        return [" " + convert_choice(doc["choice1"]), " " + convert_choice(doc["choice2"])]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        return "because" if doc["question"] == "cause" else "therefore"


class RTE(ICLMultiChoiceTaskDataset):
    """Prompt: "SENTENCE1\nQuestion: SENTENCE2 True or False?\nAnswer:"
    implement PMI_DC
    acc, random at 50% (GLUE)
    continuations: True, False

    {
        'sentence1': 'The number of Danes opposed to swapping the krone for the euro has increased slightly to 35.3 percent, up from 34.6 percent in April, according to a poll published on Thursday by Danske Bank.',
        'sentence2': 'The introduction of the euro has been opposed.',
        'label': 0,
    }
    """

    metric_type = "len_norm"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="rte",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["sentence1"] + "\nQuestion: " + doc["sentence2"] + " True or False?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" True", " False"]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class CommitmentBank(ICLMultiChoiceTaskDataset):
    """Prompt: "PREMISE\nQuestion: HYPOTHESIS. True, False or Neither?\nAnswer:"
    continuations: True, False, Neither

        implement PMI_DC
        acc/F1, random at 33% acc. (SuperGLUE)

    {
        'premise': 'Then they would awake, terrified and sweating, to find themselves in white starched linen, in a comfortable bed, in peaceful England. And all would be well. It may be said that although he survived it the siege nevertheless had a bad effect on the Collector.',
        'hypothesis': 'the siege nevertheless had a bad effect on the Collector',
        'label': 0
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="cb",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["premise"] + "\nQuestion: " + doc["hypothesis"] + ". True, False or Neither?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" True", " False", " Neither"]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class MRPC(ICLMultiChoiceTaskDataset):
    """Prompt for MRPC is formed using "Sentence 1: SENTENCE1\nSentence 2: SENTENCE2\nQuestion: Do both sentences mean the same thing?\nAnswer:"
    acc/F1, random at 50% acc. (GLUE)
    continuations: yes and no

    {
        'sentence1': 'In fiction : Edward P. Jones ( " The Known World " ) and Scott Spencer ( " A Ship Made of Paper " ) .',
        'sentence2': 'The fifth nominee for fiction is Scott Spencer , for A Ship Made of Paper .',
        'label': 0
    }
    """

    metric_type = "f1"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="mrpc",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(cls, string: str) -> str:
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return (
            "Sentence 1: "
            + self.preprocess(doc["sentence1"])
            + "\nSentence 2: "
            + self.preprocess(doc["sentence2"])
            + "\nQuestion: Do both sentences mean the same thing?\nAnswer:"
        )

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['label'] is True, return index of " yes" which is 0
        if doc["label"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class SST2(ICLMultiChoiceTaskDataset):
    """SST2 task formats prompts as "SENTENCE\nQuestion: Is this sentence positive or negative?\nAnswer:"
    some preprocessing done on sentence

    constructs 2 requests, 1 for positive and another for negative
    positive and negative have just 1 token in tokenizer
    positive: 1313
    negative: 2430

    implement PMI_DC
    acc, random at 50% (GLUE)

    {
        'sentence': "harrison 's flowers puts its heart in the right place , but its brains are in no particular place at all . ",
        'label': 1,
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="sst2",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(cls, string: str) -> str:
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return self.preprocess(doc["sentence"]) + "\nQuestion: Is this sentence positive or negative?\nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        # # {1: "positive", 0: "negative"}
        return [" negative", " positive"]

    def doc_to_label(self, doc):
        # {1: "positive", 0: "negative"}
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class MMLU(ICLMultiChoiceTaskDataset):
    """MMLU creates context with "Question: QUESTION\nAnswer:" and sends the choices as continuations
           space added as prefix to each continuation

       {
           'question': "Which of the following terms describes the body's ability to maintain its normal state?",
           'subject': 'anatomy',
           'choices': ['Anabolism', 'Catabolism', 'Tolerance', 'Homeostasis'],
    '       answer': 3
        }
    """

    metric_type = "len_norm"  # Ideally pmi_dc

    _subcategories = {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }

    _categories = {
        "stem": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social_sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other": ["other", "business", "health"],
    }

    def __init__(
        self,
        tokenizer,
        dataset_path="hails/mmlu_no_train",
        dataset_name=None,
        split="validation",
        prompt_variations=None,
        mc_labels=False,
    ):
        dataset_names = []
        # Collect the relevant categories
        if dataset_name in MMLU._categories:
            for sub_cat in MMLU._categories[dataset_name]:
                for name, cats in MMLU._subcategories.items():
                    if sub_cat in cats:
                        dataset_names.append(name)
        elif dataset_name in MMLU._subcategories:
            dataset_names.append(dataset_name)
        else:  # E.g., "math"
            for name, cats in MMLU._subcategories.items():
                if dataset_name in cats:
                    dataset_names.append(name)
        self.dev_set = {}
        self.mc_labels = mc_labels
        prompts: List[Union[None, str]] = [None]
        if prompt_variations is not None:
            if prompt_variations == 1:
                prompts = [None, "inst", "inst+1", "inst+2", "inst+3", "inst+4", "inst+5"]
            elif prompt_variations == 2:
                prompts = ["inst+5"]
            else:
                raise ValueError(f"Unknown prompt variations: {prompt_variations}")
            # Need to grab the dev set for the few-shot prompts
            for name in dataset_names:
                dev_set = load_hf_dataset(dataset_path, name, "dev")
                self.dev_set[name] = dev_set
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_names,
            split=split,
            prompts=prompts,
        )

    def doc_to_text(self, doc):
        def format_example(doc, keys):
            question_prefix = ""
            if not self.mc_labels:
                question_prefix = "Question: "  # To make context more clear
            question = question_prefix + doc["question"].strip()
            choices = ""
            if self.mc_labels:
                choices = "".join([f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])])
            prompt = f"{question}\n{choices}Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        output_text = format_example(doc, keys)

        if self.current_prompt is not None:
            prefix = ""
            if "inst" in self.current_prompt:
                subject = doc.get("subject").replace("_", " ")
                prefix = f"The following are multiple choice questions (with answers) about {subject}:\n\n"
            num_shots = re.findall("\\+(\\d+)", self.current_prompt)
            if num_shots:
                dev_set = self.dev_set.get(doc.get("subject"), [])
                num_shots_int = int(num_shots[0])
                for idx, dev_doc in enumerate(dev_set):
                    if idx >= num_shots_int:
                        break
                    if self.mc_labels:
                        answer = keys[dev_doc["answer"]]
                    else:
                        answer = dev_doc["choices"][dev_doc["answer"]]
                    prefix += format_example(dev_doc, keys) + " " + answer + "\n\n"
            output_text = prefix + output_text
        return output_text

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        if self.mc_labels:
            return [" A", " B", " C", " D"]
        return [" " + choice for choice in doc["choices"]]

    def doc_to_label(self, doc):
        return doc["answer"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class TriviaQACELoss(ICLMultiChoiceTaskDataset):
    """Sample TriviaQA entity with some fields suppressed. For CE Loss we only consider the "value"
    field as the answer to score.

    {
        'question': 'Which Lloyd Webber musical premiered in the US on 10th December 1993?',
        'question_id': 'tc_33',
        'answer': {
            'aliases': ['Sunset Blvd', ...],
            'normalized_aliases': ['sunset boulevard', ...],
            'normalized_value': 'sunset boulevard',
            'value': 'Sunset Boulevard'
        }
    }
    """

    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="trivia_qa",
        dataset_name="rc.wikipedia.nocontext",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "\nQuestion: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        return [" " + doc["answer"]["value"]]

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class NaturalQuestionsCELoss(ICLMultiChoiceTaskDataset):
    """Sample NaturalQuestions entity. For CE Loss we only consider the first answer entry to score.

    {
        'question': 'when was the last time anyone was on the moon',
        'answer': ['14 December 1972 UTC', 'December 1972']
    }
    """

    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="nq_open",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return "\nQuestion: " + doc["question"] + "\nAnswer:"

    def doc_to_continuations(self, doc):
        return [" " + doc["answer"][0]]

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class MathQA(ICLMultiChoiceTaskDataset):
    """
    Tai adds:
    {
        "Problem": "a multiple choice test consists of 4 questions , and each question has 5 answer choices . in how many r ways can the test be completed if every question is unanswered ?",
        "Rationale": "\"5 choices for each of the 4 questions , thus total r of 5 * 5 * 5 * 5 = 5 ^ 4 = 625 ways to answer all of them . answer : c .\"",
        "annotated_formula": "power(5, 4)",
        "category": "general",
        "correct": "c",
        "linear_formula": "power(n1,n0)|",
        "options": "a ) 24 , b ) 120 , c ) 625 , d ) 720 , e ) 1024"
    }
    """

    metric_type = "acc"

    def __init__(self, tokenizer, dataset_path="allenai/math_qa", dataset_name=None):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name
        )

    def doc_to_text(self, doc):
        return (
            "Question: " + doc["Problem"] + "\nAnswer:"
        )

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [" " + choice for choice in doc["options"].split(", ")]


    def doc_to_label(self, doc):
        # some doc["answerKey"] are stored as numbers
        num_to_letter = {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e"}

        if doc["correct"] in num_to_letter:
            doc["correct"] = num_to_letter[doc["correct"]]

        return ["a", "b", "c", "d", "e"].index(doc["correct"])

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class ICLGenerationTaskDataset(metaclass=abc.ABCMeta):
    """Tai adds: Base class for ICL generation tasks."""

    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset: datasets.Dataset,
        model_ctx_len: int = 2048,
        split="validation",
        prompts=[None],  # List of prompt variants to use
        stop_sequences=[],
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.model_ctx_len = model_ctx_len
        self.prompts = prompts
        self.split = split
        self.stop_sequences = stop_sequences
        self.current_prompt = None
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        # Convert stop sequences to token IDs
        self.stop_token_ids = [self.tokenizer.encode(seq, add_special_tokens=False) for seq in self.stop_sequences]

        # Prep examples
        self.samples: List[Dict[str, Any]] = []
        self.prep_examples()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        doc_id = 0
        for doc in self.dataset:
            for prompt in self.prompts:
                self.current_prompt = prompt

                doc_text = self.doc_to_text(doc)
                label_gen_id = self.token_encode(self.doc_to_label(doc))
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                        + f"\ndoc_text: {doc_text}"
                    )

                self.samples.append(
                    {
                        "doc_id": doc_id,
                        "ctx": ctx,
                        "dc_len": len(dc),
                        "query": ctx,  # No continuation for generation tasks
                        "dc_query": dc,
                        "stop_token_ids": self.stop_token_ids,
                        "label_gen_id": label_gen_id
                    }
                )

                doc_id += 1

    def pad_tokens_until_max(self, tokens, max_len=2048):
        """Truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        Queries are already truncated at max length of model_ctx_len
        This acts as additional check for all types of sequences in the batch
        """
        if len(tokens) > self.model_ctx_len:
            return tokens[-self.model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded self.model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > self.model_ctx_len
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[: self.model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        # 'ctx' and 'query' can all have variable length
        max_ctx_len = max(len(sample["ctx"]) for sample in data)
        max_dc_query_len = max(len(sample["dc_query"]) for sample in data)
        max_query_len = max(len(sample["query"]) for sample in data)

        doc_ids = []
        ctxs = []
        ctx_lens = []
        dc_lens = []
        queries = []
        dc_queries = []
        stop_token_ids = []
        label_gen_ids = []

        # pad according to max_lengths
        for sample in data:
            doc_ids.append(sample["doc_id"])

            ctx = sample["ctx"] + [self.tokenizer.pad_token_id] * (max_ctx_len - len(sample["ctx"]))
            dc_query = sample["dc_query"] + [self.tokenizer.pad_token_id] * (max_dc_query_len - len(sample["dc_query"]))

            ctxs.append(torch.LongTensor(ctx))
            dc_queries.append(torch.LongTensor(dc_query))

            ctx_lens.append(len(sample["ctx"]))
            dc_lens.append(sample["dc_len"])

            queries.append(torch.LongTensor(self.pad_tokens_until_max(sample["query"], max_len=max_query_len)))  # Since query is the same as ctx in this case
            stop_token_ids.append(sample["stop_token_ids"])  # Append stop sequences
            label_gen_ids.append(sample["label_gen_id"])  # Append stop sequences

        batch = {
            "doc_id": torch.LongTensor(doc_ids),
            "ctx": torch.stack(ctxs),
            "ctx_len": torch.LongTensor(ctx_lens),
            "dc_len": torch.LongTensor(dc_lens),
            "input_ids": torch.stack(queries),
            "dc_input_ids": torch.stack(dc_queries),
            "stop_token_ids": stop_token_ids,  # TODO: Currently having individual stop sequences to the batch, but they could share
            "label_gen_ids": label_gen_ids
        }
        return batch

    def token_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @abc.abstractmethod
    def doc_to_text(self, doc) -> str:
        """Match EAI eval harness
        returns a single context string
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_label(self, doc) -> int:
        """Match EAI eval harness
        returns continuation id which corresponds to true label
        """
        raise NotImplementedError

    def doc_to_domain_conditional(self, doc) -> str:
        """Provide string for domain conditional normalization
        by default its blank string, continuation normalized by prob conditioned on a blank
        """
        del doc
        return " "


class HendrycksMath(ICLGenerationTaskDataset):
    """
    Tai adds: HendrycksMath Dataset includes 8 sub types.
    {
        'problem': 'A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.',
        'level': 'Level 1',
        'type': 'Counting & Probability',
        'solution': 'The spinner is guaranteed to land on exactly one of the three regions, so we know that the sum of the probabilities of it landing in each region will be 1. If we let the probability of it landing in region $C$ be $x$, we then have the equation $1 = \\frac{5}{12}+\\frac{1}{3}+x$, from which we have $x=\\boxed{\\frac{1}{4}}$.'
    }
    """
    metric_type = "exact_match_hendrycks_math"
    DATASET_NAMES = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

    def __init__(self, tokenizer, sample_size=250, dataset_path="hendrycks/competition_math", dataset_name=None, split="test", stop_sequences=["Problem:"]):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        dataset = self.load_dataset(dataset_name, split)
        super().__init__(tokenizer, dataset, model_ctx_len=2048, split=split, prompts=[None], stop_sequences=stop_sequences)

    def load_dataset(self, dataset_name, split):
        dataset_names = self._get_dataset_names(dataset_name)
        datasets_dict = {}

        url = "https://huggingface.co/datasets/hendrycks/competition_math/resolve/main/data/MATH.zip"
        data_dir = "data_hendrycks"

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(data_dir)

        for name in dataset_names:
            basepath = os.path.join(data_dir, "MATH", split, name)
            datasets_dict[name] = self._generate_examples(basepath)

        combined_dataset = datasets.concatenate_datasets(list(datasets_dict.values()))
        return combined_dataset

    def _get_dataset_names(self, dataset_name):
        if dataset_name:
            if dataset_name in self.DATASET_NAMES:
                return [dataset_name]
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")
        return self.DATASET_NAMES

    def _generate_examples(self, basepath):
        examples = []
        json_paths = sorted(pathlib.Path(basepath).iterdir())[:self.sample_size]
        for file in json_paths:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                examples.append({
                    "problem": data["problem"],
                    "level": data["level"],
                    "type": data["type"],
                    "solution": data["solution"],
                })
        return datasets.Dataset.from_list(examples)

    def doc_to_text(self, doc):
        return (
            f"Problem: {doc['problem']}\nAnswer: {doc['solution']}"
        )

    def doc_to_label(self, doc):
        return doc["solution"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


label_to_task_map = {
    "piqa": PIQA,
    "hellaswag": HellaSwag,
    "winogrande": WinoGrande,
    "openbook_qa": OpenBookQA,
    "boolq": BoolQ,
    "sciq": SciQ,
    "arc_easy": ArcEasy,
    "arc_easy_ppl": ArcEasyCELoss,
    "arc_challenge": ArcChallenge,
    "basic_arithmetic": BasicArithmetic,
    "copa": COPA,
    "rte": RTE,
    "commitment_bank": CommitmentBank,
    "mrpc": MRPC,
    "sst2": SST2,
    "commonsense_qa": CommonsenseQA,
    "social_iqa": SocialIQa,
    "trivia_qa_wiki_ppl": TriviaQACELoss,
    "natural_qs_open_ppl": NaturalQuestionsCELoss,
    "mmlu_stem_test": (MMLU, {"dataset_name": "stem", "split": "test"}),
    "mmlu_humanities_test": (MMLU, {"dataset_name": "humanities", "split": "test"}),
    "mmlu_social_sciences_test": (MMLU, {"dataset_name": "social_sciences", "split": "test"}),
    "mmlu_other_test": (MMLU, {"dataset_name": "other", "split": "test"}),
    "mmlu_stem": (MMLU, {"dataset_name": "stem"}),
    "mmlu_humanities": (MMLU, {"dataset_name": "humanities"}),
    "mmlu_social_sciences": (MMLU, {"dataset_name": "social_sciences"}),
    "mmlu_other": (MMLU, {"dataset_name": "other"}),
    "mmlu_stem_var": (MMLU, {"dataset_name": "stem", "prompt_variations": 1}),
    "mmlu_humanities_var": (MMLU, {"dataset_name": "humanities", "prompt_variations": 1}),
    "mmlu_social_sciences_var": (MMLU, {"dataset_name": "social_sciences", "prompt_variations": 1}),
    "mmlu_other_var": (MMLU, {"dataset_name": "other", "prompt_variations": 1}),
    "mmlu_stem_mc_5shot": (MMLU, {"dataset_name": "stem", "prompt_variations": 2, "mc_labels": True}),
    "mmlu_humanities_mc_5shot": (MMLU, {"dataset_name": "humanities", "prompt_variations": 2, "mc_labels": True}),
    "mmlu_social_sciences_mc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_other_mc_5shot": (MMLU, {"dataset_name": "other", "prompt_variations": 2, "mc_labels": True}),
    "mmlu_stem_mc_5shot_test": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_humanities_mc_5shot_test": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_social_sciences_mc_5shot_test": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),
    "mmlu_other_mc_5shot_test": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompt_variations": 2, "mc_labels": True},
    ),

    # Tai: Adds Math tasks
    "mathqa": MathQA,
    "math_algebra": (HendrycksMath, {"dataset_name": "algebra", "stop_sequences": HENDRYCKS_STOP_SEQS}),
    "math_counting_and_prob": (HendrycksMath, {"dataset_name": "counting_and_probability", "stop_sequences": HENDRYCKS_STOP_SEQS}),
    "math_geometry": (HendrycksMath, {"dataset_name": "geometry", "stop_sequences": HENDRYCKS_STOP_SEQS}),
    "math_intermediate_algebra": (HendrycksMath, {"dataset_name": "intermediate_algebra", "stop_sequences": HENDRYCKS_STOP_SEQS}),
    "math_num_theory": (HendrycksMath, {"dataset_name": "number_theory", "stop_sequences": HENDRYCKS_STOP_SEQS}),
    "math_prealgebra": (HendrycksMath, {"dataset_name": "prealgebra", "stop_sequences": HENDRYCKS_STOP_SEQS}),
    "math_precalc": (HendrycksMath, {"dataset_name": "precalculus", "stop_sequences": HENDRYCKS_STOP_SEQS})
}
