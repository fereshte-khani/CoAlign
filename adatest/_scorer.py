import numpy as np
import re
import logging
import uuid
import itertools
import sentence_transformers
import openai
# from openai.embeddings_utils import get_embedding, cosine_similarity

#from transformers.tokenization_utils_base import ExplicitEnum
#from ._explorer import file_log

log = logging.getLogger(__name__)

class Scorer():
    pass

# <option>should not be</option>
#                     <option>should be</option>
#                     <option>should be the same as for</option>
#                     <option>should not be less than for</option>

class DummyScorer(Scorer):
    def __init__(self):
        self._id = uuid.uuid4().hex
    def __call__(self, tests):
        out = []
        for k, test in tests.iterrows():
            try:
                score = float(test.value2)
            except:
                score = np.nan
            out.append(score)
        return np.array(out)

class ClassifierScorer(Scorer):
    """ Wraps a model and defines a callable scorer that returns a score value for any input/output pair.

    For example if we wrap a text sentiment classifer the scorer("I love this movie!", "Positive") will return
    a large positive value indicating that the model is very likely to produce that output when given that
    input.
    """

    def __init__(self, model, topk=1):
        self._id = uuid.uuid4().hex
        self.model = model
        self.output_names = self.model.output_names
        if not callable(self.output_names):
            self._output_name_to_index = {v: i for i, v in enumerate(self.output_names)}
        self.topk = topk
        self.output_type = "classification"

    def __call__(self, tests, goal_embedding=None):
        if self.output_type == "classification":
            eval_inputs = []
            eval_inds = []
            variations1 = []
            variations2 = []
            for i, (k, test) in enumerate(tests.iterrows()):
                if test.comparator == "should not be" or test.comparator == "should be":
                    v1 = expand_template(test.value1)
                    for s1 in v1:
                        eval_inputs.append(s1)
                        eval_inds.append(i)
                    variations1.append(v1)
                    variations2.append(None)
                elif test.comparator == "should be the same as for":
                    # eval_inputs.append(test.value1)
                    # eval_inputs.append(test.value2)
                    v1 = expand_template(test.value1)
                    v2 = expand_template(test.value2)
                    for s1 in v1:
                        for s2 in v2:
                            eval_inputs.append(s1)
                            eval_inputs.append(s2)
                            eval_inds.append(i)
                            eval_inds.append(i)
                    variations1.append(v1)
                    variations2.append(v2)

            try:
                model_out = self.model(eval_inputs)
            except Exception as e:
                model_out = np.zeros((len(eval_inputs), len(self.model.output_names))) * np.nan # TODO: remove this hack after the user study
                log.error(e)
                log.error(eval_inputs)
                log.error("The model threw an exception when evaluating inputs! We are patching this disaster with np.nan for the sake of the user study!")

            out = [[] for _ in range(tests.shape[0])]
            out_pos = 0
            i = 0
            value1_outputs = [{} for _ in range(tests.shape[0])]
            value2_outputs = [{} for _ in range(tests.shape[0])]
            while i < len(model_out):
                out_pos = eval_inds[i]

                comparator = tests.iloc[out_pos]["comparator"]
                if comparator == "should not be" or comparator == "should be":

                    # save the top model outputs
                    inds = np.argsort(-model_out[i])
                    shown_tmp = {}
                    for j in inds[:5]:
                        shown_tmp[self.model.output_names[j]] = float(model_out[i][j])
                    value1_outputs[out_pos] = shown_tmp

                    token_to_check = tests.iloc[out_pos]['value2']

                    # TODO: This is a hack where we're looking for different capitalizations of the output if the original one doesn't exist
                    # we added this because of gpt-2 word splitting (which makes 'muslim' not be in the output)
                    # we should fix this at some point :P
                    if token_to_check not in self._output_name_to_index:
                        if token_to_check.capitalize() in self._output_name_to_index:
                            token_to_check = token_to_check.capitalize()
                        elif token_to_check.lower() in self._output_name_to_index:
                            token_to_check = token_to_check.lower()
                    
                    # multiple tokens can be checked at the same time with templates
                    out_val = np.nan
                    for token_part in expand_template(token_to_check):
                        ind = self._output_name_to_index.get(token_part, None)
                        if ind is not None and model_out[i] is not None:
                            sorted_values = np.argsort(model_out[i])
                            topk = topk_threshold_ind(ind, sorted_values, self.topk)
                            if np.isnan(model_out[i][ind]):
                                score = np.nan
                            elif model_out[i][ind] > model_out[i][topk]:
                                score = model_out[i][ind] - model_out[i][topk]
                            else:
                                mask = (model_out[i] <= model_out[i][topk]) & (model_out[i] > model_out[i][ind])
                                score = (model_out[i][ind] - model_out[i][mask]).sum()
                            if comparator == "should be":
                                score *= -1
                            # out_val = max(score, out_val)
                            out[out_pos].append(score)
                    # out[out_pos] = max(out[out_pos], out_val)
                    i += 1
                elif comparator == "should be the same as for":

                    # save the top model outputs


                    inds = np.argsort(-model_out[i])
                    shown_tmp = {}
                    for j in inds[:5]:
                        shown_tmp[self.model.output_names[j]] = float(model_out[i][j])
                    value1_outputs[out_pos] = shown_tmp
                    inds = np.argsort(-model_out[i+1])
                    shown_tmp = {}
                    for j in inds[:5]:
                        shown_tmp[self.model.output_names[j]] = float(model_out[i+1][j])
                    value2_outputs[out_pos] = shown_tmp
                    
                    score = equality_score(model_out[i], model_out[i+1])
                    # out[out_pos] = max(out[out_pos], score)
                    out[out_pos].append(score)
                    i += 2
                else:
                    raise Exception(f"Comparator type '{comparator}' not yet supported!")

                # out_pos += 1
            return out, value1_outputs, value2_outputs
        else:
            raise Exception(f"Output type {self.output_type} not yet supported!")

    def suggest_outputs(self, current, num_suggestions=20):
        prompt = ""
        for c in current:
            prompt += '"'+c+'"\n'
        prompt += '"{output}'
        response = openai.Completion.create(
            engine='curie-instruct-beta', prompt=[prompt.format(output=o) for o in self.output_names], max_tokens=0, # self.engine
            temperature=0, n=1, stop='\"', logprobs=0, echo=True
        )
        lines = [sum(choice["logprobs"]["token_logprobs"][11:]) for choice in response["choices"]]
        pairs = list([v for v in zip(lines, self.output_names) if v[1] not in current])
        pairs.sort()
        return [v[1] for v in list(reversed(pairs))[:num_suggestions]]
        
TextScorer = ClassifierScorer

# class GeneratorScorer(Scorer):
#     """ Wraps a model and defines a callable scorer that returns a score value for any input/output pair.
#     """

#     def __init__(self, model):
#         self._id = uuid.uuid4().hex
#         self.model = model

#     def __call__(self, tests):
#         eval_inputs = []
#         eval_inds = []
#         variations1 = []
#         variations2 = []
#         for i, (k, test) in enumerate(tests.iterrows()):
#             if test.comparator == "should not be" or test.comparator == "should be":
#                 v1 = expand_template(test.value1)
#                 for s1 in v1:
#                     eval_inputs.append(s1)
#                     eval_inds.append(i)
#                 variations1.append(v1)
#                 variations2.append(None)
#             elif test.comparator == "should be the same as for":
#                 # eval_inputs.append(test.value1)
#                 # eval_inputs.append(test.value2)
#                 v1 = expand_template(test.value1)
#                 v2 = expand_template(test.value2)
#                 for s1 in v1:
#                     for s2 in v2:
#                         eval_inputs.append(s1)
#                         eval_inputs.append(s2)
#                         eval_inds.append(i)
#                         eval_inds.append(i)
#                 variations1.append(v1)
#                 variations2.append(v2)

#         try:
#             model_out = self.model(eval_inputs)
#         except Exception as e:
#             model_out = ["ERROR" for _ in range(len(eval_inputs))]#np.zeros((len(eval_inputs), len(self.model.output_names))) * np.nan # TODO: remove this hack after the user study
#             log.error(e)
#             log.error(eval_inputs)
#             log.error("The model threw an exception when evaluating inputs! We are patching this disaster with 'ERROR' for the sake of the user study!")

#         out = [[] for _ in range(tests.shape[0])]
#         out_pos = 0
#         i = 0
#         value1_outputs = [{} for _ in range(tests.shape[0])]
#         value2_outputs = [{} for _ in range(tests.shape[0])]
#         while i < len(model_out):
#             out_pos = eval_inds[i]

#             comparator = tests.iloc[out_pos]["comparator"]
#             if comparator == "should not be" or comparator == "should be":
                
#                 # auto fill missing outputs
#                 if tests.iloc[out_pos]['value2'] is None:
#                     tests.iloc[out_pos]['value2'] = model_out[i]
                
#                 # save the model output
#                 value1_outputs[out_pos]  = {}
#                 value1_outputs[out_pos][model_out[i]] = 1

#                 # multiple tokens can be checked at the same time with templates
#                 for token_part in expand_template(tests.iloc[out_pos]['value2']):
#                     out[out_pos].append(1 if model_out[i] == token_part else -1)
#                 i += 1
#             elif comparator == "should be the same as for":

#                 # save the model outputs
#                 value1_outputs[out_pos]  = {}
#                 value1_outputs[out_pos][model_out[i]] = 1
#                 value2_outputs[out_pos]  = {}
#                 value2_outputs[out_pos][model_out[i+1]] = 1
                
#                 # save the score
#                 out[out_pos].append(1 if model_out[i] == model_out[i+1] else -1)
#                 i += 2
#             else:
#                 raise Exception(f"Comparator type '{comparator}' not yet supported!")

#             # out_pos += 1
#         return out, value1_outputs, value2_outputs


class GeneratorScorer(Scorer):
    """ Wraps a model and defines a callable scorer that returns a score value for any input/output pair.
    """

    def __init__(self, model, reverse_model=None, embedding_model=None, similarity_threshold=0.9):
        self._id = uuid.uuid4().hex
        self.model = model
        self.reverse_model = reverse_model
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def __call__(self, tests):

        # run the model on the inputs
        eval_inputs = []
        eval_inds = []
        eval_reverse_pos = []
        variations1 = []
        variations2 = []
        for i, (k, test) in enumerate(tests.iterrows()):
            if test.comparator == "should not be" or test.comparator == "should be":
                v1 = expand_template(test.value1)
                for s1 in v1:
                    eval_inputs.append(s1)
                    eval_inds.append(i)
                variations1.append(v1)
                variations2.append(None)
            elif test.comparator == "should be invertable.":
                v1 = expand_template(test.value1)
                for s1 in v1:
                    eval_inputs.append(s1)
                    eval_inds.append(i)
                    eval_reverse_pos.append(len(eval_inputs) - 1)
            elif test.comparator == "should be the same as for":
                v1 = expand_template(test.value1)
                v2 = expand_template(test.value2)
                for s1 in v1:
                    for s2 in v2:
                        eval_inputs.append(s1)
                        eval_inputs.append(s2)
                        eval_inds.append(i)
                        eval_inds.append(i)
                variations1.append(v1)
                variations2.append(v2)
        try:
            model_out = self.model(eval_inputs)
        except Exception as e:
            model_out = ["ERROR" for _ in range(len(eval_inputs))]#np.zeros((len(eval_inputs), len(self.model.output_names))) * np.nan # TODO: remove this hack after the user study
            log.error(e)
            log.error(eval_inputs)
            log.error("The model threw an exception when evaluating inputs! We are patching this disaster with 'ERROR' for the sake of the user study!")

        # run the reverse model on any outputs we need to
        # eval_reverse_inputs = []
        
        # for i, (k, test) in enumerate(tests.iterrows()):
        #     if test.comparator == "should be invertable.":
        #         v1 = expand_template(test.value1)
        #         for s1 in v1:
        #             eval_reverse_inputs.append(s1)
        #             eval_reverse_inds.append(i)
        if len(eval_reverse_pos) > 0:
            model_reverse_out = [None for _ in model_out]
            input_embed = [None for _ in model_out]
            round_trip_embed = [None for _ in model_out]
            try:
                # compute input embedding
                tmp = self.embedding_model.encode([eval_inputs[ind] for ind in eval_reverse_pos], convert_to_tensor=True, show_progress_bar=False).cpu()
                for i, ind in enumerate(eval_reverse_pos):
                    input_embed[ind] = tmp[i]

                # compute reverse model output
                reverse_out = self.reverse_model([model_out[ind] for ind in eval_reverse_pos])
                for i, ind in enumerate(eval_reverse_pos):
                    model_reverse_out[ind] = str(reverse_out[i])

                # compute round trip embedding
                tmp = self.embedding_model.encode(reverse_out, convert_to_tensor=True, show_progress_bar=False).cpu()
                for i, ind in enumerate(eval_reverse_pos):
                    round_trip_embed[ind] = tmp[i]

            except Exception as e:
                model_reverse_out = ["ERROR" for _ in range(len(model_out))]
                log.error(e)
                log.error("The reverse model threw an exception when evaluating inputs! We are patching this disaster with 'ERROR' for the sake of the user study!")
        else:
            model_reverse_out = []

        out = [[] for _ in range(tests.shape[0])]
        out_pos = 0
        i = 0
        value1_outputs = [{} for _ in range(tests.shape[0])]
        value2_outputs = [{} for _ in range(tests.shape[0])]
        while i < len(model_out):
            out_pos = eval_inds[i]

            comparator = tests.iloc[out_pos]["comparator"]
            if comparator == "should not be" or comparator == "should be":
                
                # auto fill missing outputs
                if tests.iloc[out_pos]['value2'] is None:
                    tests.loc[tests.index[out_pos], 'value2'] = str(model_out[i])
                
                # save the model output
                value1_outputs[out_pos]  = {}
                value1_outputs[out_pos][model_out[i]] = 1

                # multiple tokens can be checked at the same time with templates
                for token_part in expand_template(tests.iloc[out_pos]['value2']):
                    out[out_pos].append(1 if model_out[i] == token_part else -1)
                i += 1
            elif comparator == "should be invertable.":
                
                # compare embedding distances
                score = sentence_transformers.util.pytorch_cos_sim(input_embed[i], round_trip_embed[i]).numpy()[0][0]
                out[out_pos].append(self.similarity_threshold-score)

                # update the output since it is always computed in inversion tests
                tests.loc[tests.index[out_pos], 'value2'] = str(model_reverse_out[i])
                
                # save the model round trip output
                value1_outputs[out_pos]  = {}
                value1_outputs[out_pos][str(model_out[i])] = 1

                i += 1
            elif comparator == "should be the same as for":

                # save the model outputs
                value1_outputs[out_pos]  = {}
                value1_outputs[out_pos][model_out[i]] = 1
                value2_outputs[out_pos]  = {}
                value2_outputs[out_pos][model_out[i+1]] = 1
                
                # save the score
                out[out_pos].append(1 if model_out[i] == model_out[i+1] else -1)
                i += 2
            else:
                raise Exception(f"Comparator type '{comparator}' not yet supported!")

            # out_pos += 1
        return out, value1_outputs, value2_outputs


def expand_template(s):
    """ Expand a template string into a list of strings.
    """
    # parts = []
    # for s in strings:
    matches = re.findall("{[^}]*}", s)
    s = re.sub("{[^}]*}", "{}", s)
    template_groups = [str(m)[1:-1].split("|") for m in matches]
    try:
        return [s.format(*parts) for parts in itertools.product(*template_groups)]
    except ValueError:
        return [s] # we return the template not filled in if it is invalid

def clean_template(s):
    """ This removes duplicate template entries.
    """
    matches = re.findall("{[^}]*}", s)
    s = re.sub("{[^}]*}", "{}", s)
    template_groups = [str(m)[1:-1].split("|") for m in matches]
    clean_groups = ["{"+"|".join(list({v: None for v in g}.keys()))+"}" for g in template_groups]
    try:
        return s.format(*clean_groups)
    except ValueError:
        return s # we return the template not cleaned in if it is invalid

def topk_threshold_ind(ind, sorted_values, k):
    """ Return the threshold value for which if ind dropped below it would not be in the top k (without other scores changing).
    """
    if ind in sorted_values[-k:]:
        topk = sorted_values[-k - 1]
    else:
        topk = sorted_values[-k]
    if topk == ind:
        topk = sorted_values[-k - 1]
    return topk


def equality_score(output_values1, output_values2, topk=1):
    assert topk == 1
    ind1 = np.argmax(output_values1)
    ind2 = np.argmax(output_values2)
    max1 = output_values1[ind1]
    max2 = output_values2[ind2]
    margins = np.zeros(len(output_values1))

    if ind1 != ind2:
        min_margin = 1e6
        for i in range(len(output_values1)):
            score1 = max(0, max1 - output_values1[i])
            score2 = max(0, max2 - output_values2[i])
            margin = score1 + score2
            if margin < min_margin:
                min_margin = margin
        return min_margin
    else:
        val1 = output_values1[ind1]
        output_values1[ind1] = np.nan
        score1 = val1 - np.nanmax(output_values1)

        val2 = output_values2[ind2]
        output_values2[ind2] = np.nan
        score2 = val2 - np.nanmax(output_values2)
        return -min(score1, score2)
