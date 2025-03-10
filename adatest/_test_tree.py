from typing import TextIO
import warnings
from IPython.display import display, Markdown, clear_output, HTML
from numpy.lib.function_base import select
from numpy.lib.npyio import savez_compressed
import openai
import numpy as np
import random
import copy
import sentence_transformers
import pandas as pd
import torch
import json
import re
import functools
import collections
from .comm import JupyterComm
import time
import threading
import uuid
from ._scorer import TextScorer, Scorer
from types import MethodType
import pathlib
import sys
import copy
import traceback
import re
import logging
import os
import io
import statistics
# import checklist.editor
from threading import Timer
from ._scorer import expand_template, clean_template
import adatest
from coalign.utils.test_tree_functions import adding_new_sentences_to_test_tree, generate_fine_tuned_models_of_a_file, launching_browser
import coalign.src.coalign_code_config

import coalign.src.coalign_code_config as ddubg_code_config
# from openai.embeddings_utils import get_embedding, cosine_similarity


# from ttps://gist.github.com/walkermatt/2871026
def debounce(wait):
    """ Decorator that will postpone a functions
        execution until after wait seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        def debounced(*args, **kwargs):
            def call_it():
                fn(*args, **kwargs)
            try:
                debounced.t.cancel()
            except(AttributeError):
                pass
            debounced.t = Timer(wait, call_it)
            debounced.t.start()
        return debounced
    return decorator

log = logging.getLogger(__name__)

# import sys
# sys.stderr = open('/tmp/err.txt', 'w')

def file_log(*args):
    """ Used for logging when we don't have a stdout.

    This is used for debugging when we are being called from the javascript client. When we are
    called from the client we don't have a stdout attached to any cell in the notebook.

    Note to also catch errors you could do:
    import sys
    sys.stderr = open('err.txt', 'w')
    """
    #print(*args)
    f = open("log.txt", "a")  # append mode
    f.write(" ".join([str(msg) for msg in args])+"\n")
    f.flush()
    f.close()

cached_embedding_model = None

def is_subtopic(topic, candidate):
    # Returns true if candidate is a subtopic of topic
    return True if re.search(r'^%s(/|$)' % topic.replace('+', r'\+'), candidate) else False

special_outputs = [
    "{MAX}"
]

valid_comparators = [
    "should not be",
    "should be",
    "should be the same as for"
]
FILLIN_PREFIX = '/Fill-ins'


# model("this is english") => []
# output_sampling="topk(10)"
# output_sampling="topp(10)"
# output_sampling="max"
# output_sampling="temperature(0.9)"

class TestTreeBrowser():
    """ This represents what we get when we call a TestTree Object.

    TODO: Factor more stuff into this object from TestTree.
    """
#FK I added the check_conflict flag & predictor
    def __init__(self, test_tree, scorer, starting_topic, max_suggestions, max_suggestions_display, slot_randomization,
                 score_randomization, skip_randomization, prompt_size, complete_diversity, prompt_diversity,
                 use_focus, focus_decay, prompt_threads, temperature, subtopic_diversity, score_filter, experiment,
                 embedding_model, prompt_seperator, user, recompute_scores, backend, topic_model_scale, generate_outputs,
                 drop_inactive_scores, predictor = None, check_conflict=False, task=None):

        self.test_tree = test_tree
        self.scorer = scorer
        self.current_topic = starting_topic
        self.max_suggestions = max_suggestions
        self.max_suggestions_display = max_suggestions_display
        self.slot_randomization = slot_randomization
        self.score_randomization = score_randomization
        self.skip_randomization = skip_randomization
        self.prompt_size = prompt_size
        self.complete_diversity = complete_diversity
        self.prompt_diversity = prompt_diversity
        self.use_focus = use_focus
        self.focus_decay = focus_decay
        self.prompt_threads = prompt_threads
        self.temperature = temperature
        self.subtopic_diversity = subtopic_diversity
        self.score_filter = score_filter
        self.experiment = experiment
        self.prompt_seperator = prompt_seperator
        self.user = user # who is doing the labeling session
        self.backend = backend
        self.topic_model_scale = topic_model_scale
        self.generate_outputs = generate_outputs
        self.check_conflict = check_conflict
        self.predictor = predictor
        self.task = task

        if self.backend is None:
            self.backend = adatest.backend

        # if we are recomputing the scores then we erase all the old ones
        if recompute_scores is True:
            for c in self.test_tree.columns:
                if c.endswith(" score") or c == "score":
                    self.test_tree.drop(c, axis=1, inplace=True)

        # find the first (and hence main) score
        if isinstance(self.scorer, dict):
            self.score_columns = [c+" score" for c in self.scorer]
            for c in self.score_columns:
                if c not in self.test_tree.columns:
                    self.test_tree[c] = [np.nan for _ in range(self.test_tree.shape[0])]
        else:
            self.score_columns = []
        for c in self.test_tree.columns:
            if c.endswith("score") and c not in self.score_columns:
                if drop_inactive_scores is True:
                    self.test_tree.drop(c, axis=1, inplace=True)
                else:
                    self.score_columns.append(c)

        self._id = uuid.uuid4().hex # a unique identifier for this test set instance

        # these are all temporary state
        self._embeddings = {} # Cached embedding vectors for each input/output pair (keyed on the pair ids)

        self._hidden_topics = {}
        self.comm = None

        self._checklist_tester = None


        # set the embedding model we use for similarity computations
        global cached_embedding_model
        log.debug(f"passed embedding_model {embedding_model}")
        if self.scorer is not None and embedding_model is None:
            if cached_embedding_model is None:
                cached_embedding_model = sentence_transformers.SentenceTransformer('stsb-roberta-large')
            self.embedding_model = cached_embedding_model
        else:
            self.embedding_model = embedding_model

        self._compute_embeddings_and_scores(self.test_tree)

        # init a blank set of suggetions
        self.suggestions = pd.DataFrame([], columns=self.test_tree.columns)
        self.suggestions_page_position = 0
        self.suggestions = self._ensure_add_item_row(self.suggestions)
        self.suggestions_error = False # tracks if we failed to generate suggestions

    def _repr_html_(self, prefix="", comm="jupyter", environment="jupyter", websocket_server=None):
        """ Returns the HTML interface for this browser.
        """

        # if self.comm is not None:
        #     self.comm.send({"disabled": True})
        if comm is not None:
            self.comm = comm if comm != "jupyter" else JupyterComm(f'gadfly_interface_target_{self._id}', self.update)

        # dump the client javascript to the interface
        file_path = pathlib.Path(__file__).parent.absolute()
        with open(file_path / ".." / "client" / "dist" / "main.js", encoding="utf-8") as f:
            js_data = f.read()
        interface_html = f"""
<div id="gadfly_container_{self._id}" style="width: 100%; all: initial;"></div>
<script src="https://kit.fontawesome.com/fcd9b03029.js" crossorigin="anonymous"></script>
<script crossorigin src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
<script type='text/javascript'>
  {js_data};
  ReactDOM.render(
    React.createElement(Gadfly, {{
      interfaceId: "{self._id}", environment: "{environment}", startingTopic: "{self.current_topic}", prefix: "{prefix}",
      websocket_server: {"undefined" if websocket_server is None else '"'+websocket_server+'"'},
      checklistMode: {"true" if self.experiment is not None and self.experiment.get("checklist_mode", False) else "false"}
    }}, null),
    document.getElementById('gadfly_container_{self._id}')
  );
</script>
"""
        return interface_html

    def display(self):
        """ Manually display the HTML interface.
        """
        display(HTML(self._repr_html_()))


    def update(self, msg):
        log.debug(f"update({msg})")

        # try:
        to_send = {}
        for k in msg:
            echo = False

            # an action specific to a row
            if k != "test_chart" and msg[k].get("action", None) is not None:
                df = None
                if k in self.suggestions.index:
                    df = self.suggestions
                elif k in self.test_tree.index:
                    df = self.test_tree

                if msg[k]["action"] == "template_expand_value1":
                    template_value = self.templatize(df.loc[k, "value1"])
                    msg[k] = {"value1": template_value} # we convert this to a standard value1 update
                    echo = True
                else:
                    log.debug(f"unknown row action: {msg[k]['action']}")

            if k == "test_chart" and msg[k].get("action", None) is not None:
                if msg[k]["action"] == "button1":
                    print('button 1 is pressed')

                    print('the current topic is ', self.current_topic)
                    generate_fine_tuned_models_of_a_file(self.task, 
                                                         self.test_tree, 
                                                         self.current_topic, 
                                                         ddubg_code_config.original_model_path, 
                                                        ddubg_code_config.local_model_path, 
                                                        num_epochs=ddubg_code_config.num_epochs)
                    tokenizer, local_model = self.task.upload_pretrained_model(ddubg_code_config.local_model_path) 
                    local_predictor, local_scorer = self.task.get_model_and_scorer(local_model, tokenizer)
                    self.predictor = local_predictor
                    print('local_predictor is loaded')
                    self._finish_updating()
                    print('finish updating')

                elif msg[k]["action"] == "button2":
                    print('button 2 is pressed')
                    generate_fine_tuned_models_of_a_file(self.task, self.test_tree, None, 
                                                        ddubg_code_config.original_model_path, 
                                                        ddubg_code_config.global_model_path, 
                                                        num_epochs=ddubg_code_config.num_epochs)
                    tokenizer, global_model = self.task.upload_pretrained_model(ddubg_code_config.global_model_path) 
                    global_predictor, global_scorer = self.task.get_model_and_scorer(global_model, tokenizer)
                    self.scorer = global_scorer
                    print('dropping scores first')
                    # for c in self.test_tree.columns:
                    #     if c.endswith(" score") or c == "score":
                    #         self.test_tree.drop(c, axis=1, inplace=True)
                    self._compute_scores(self.test_tree, recompute=True)
                    print('done computing scores')
                    self._finish_updating()
                    self._update_interface()
                    print('global scorer is loaded')

                elif msg[k]["action"] == "redraw":
                    self._update_interface()
                elif msg[k]["action"] == "refresh_suggestions":
                    try:
                        self.suggestions = self._generate_suggestions(
                            self.current_topic,
                            value1_filter=msg[k].get("value1_filter", None),
                            comparator_filter=msg[k].get("comparator_filter", None),
                            value2_filter=msg[k].get("value2_filter", None),
                            suggestions_template_value1=msg[k].get("suggestions_template_value1", None), # TODO: this is a hack for the checklist baseline case
                            suggestions_template_comparator=msg[k].get("suggestions_template_comparator", None),
                            suggestions_template_value2=msg[k].get("suggestions_template_value2", None),
                            checklist_mode=msg[k].get("checklist_mode", False)
                        )

                        self.suggestions.sort_values(self.score_columns[0], inplace=True, ascending=False, key=np.vectorize(score_max))
                        self.suggestions_error = False
                    except openai.error.APIError as e:
                        log.debug(e)
                        self.suggestions = pd.DataFrame([], columns=self.test_tree.columns)
                        self.suggestions_page_position = 0
                        self.suggestions = self._ensure_add_item_row(self.suggestions)
                        self.suggestions_error = True
                    self.suggestions_page_position = 0
                    #self.suggestions = self.all_suggestions.iloc[:self.max_suggestions_display]
                    self._update_interface()
                elif msg[k]["action"] == "change_topic":
                    self.current_topic = msg[k]["topic"]
                    self.suggestions = pd.DataFrame([], columns=self.test_tree.columns)
                    self.suggestions = self._ensure_add_item_row(self.suggestions)
                    self._update_interface()
                elif msg[k]["action"] == "clear_suggestions":
                    log.debug("clearing suggetions")
                    self.suggestions = pd.DataFrame([], columns=self.test_tree.columns)
                    self.suggestions = self._ensure_add_item_row(self.suggestions)
                    self._update_interface()
                elif msg[k]["action"] == "add_new_topic":
                    log.debug("add_new_topic")
                    self.suggestions = pd.DataFrame([], columns=self.test_tree.columns)
                    self._update_interface()
                elif msg[k]["action"] == "set_first_model":
                    log.debug("set_first_model")
                    name = msg[k]["model"]

                    # move to front of score columns in data frames
                    for df in [self.suggestions, self.test_tree]:
                        pos = len(df.columns) - len(self.score_columns)
                        tmp = df[name]
                        df.drop(labels=[name], axis=1, inplace=True)
                        df.insert(pos, name, tmp)

                    # update score columns list
                    self.score_columns.remove(name)
                    self.score_columns.insert(0, name)

                    self._update_interface()

            elif k == "test_chart" and msg[k].get("engine", None) is not None:
                self.engine = msg[k]["engine"]

            # if we are updating a row in suggestions or tests then we recompute the scores
            elif "hidden" in msg[k] and len(msg[k]) == 1:
                self._hidden_topics[k] = msg[k]["hidden"]
                self.comm.send({k: {"hidden": self._hidden_topics[k]}})
            elif "topic" not in msg[k]:
                df = None
                if k in self.suggestions.index:
                    log.debug(f"found k in self.suggestions")
                    df = self.suggestions
                elif k in self.test_tree.index:
                    df = self.test_tree
                if df is not None:
                    for k2 in msg[k]:
                        df.loc[k, k2] = msg[k][k2]
                    if self._is_fillin():
                        df.loc[k, self.score_columns] = np.nan
                    else:
                        df.loc[k, self.score_columns] = None
                        self._compute_embeddings_and_scores(df)
                    data = {k: {
                        "scores": {c: [[k, v] for v in score_parts(df.loc[k, c])] for c in self.score_columns},
                        "comparator": df.loc[k, "comparator"],
                        "value2": df.loc[k, "value2"],
                        "value1_outputs": {c: [[k, json.loads(df.loc[k].get(c + " value1 outputs", "{}"))]] for c in self.score_columns},
                        "value2_outputs": {c: [[k, json.loads(df.loc[k].get(c + " value2 outputs", "{}"))]] for c in self.score_columns},
                    }}
                    if echo:
                        for k2 in msg[k]:
                            data[k][k2] = msg[k][k2]
                    log.debug(f"self.comm.send({data})")
                    self.comm.send(data)

            # if we are just changing the topic
            elif "topic" in msg[k] and len(msg[k]) == 1:
                if k in self.suggestions.index:
                    self.suggestions.loc[k, "topic"] = msg[k]["topic"]
                    self.suggestions.loc[k, "labeler"] = self.user
                    self.test_tree.loc[k] = self.suggestions.loc[k]
                    self.suggestions.drop(k, inplace=True)
                    self.suggestions = self._ensure_add_item_row(self.suggestions)
                elif k in self.test_tree.index:
                    if msg[k]["topic"] == "DO_DELETE__djk39sd": # this means delete the test
                        self.test_tree.drop(k, inplace=True)
                    elif msg[k]["topic"] == "suggestion": # this means move the test back to the suggestions list
                        self.test_tree.loc[k, "topic"] = msg[k]["topic"]
                        self.suggestions.loc[k] = self.test_tree.loc[k]
                        self.test_tree.drop(k, inplace=True)
                    else:
                        self.test_tree.loc[k, "topic"] = msg[k]["topic"]
                        self.test_tree.loc[k, "labeler"] = self.user
                else:
                    for id, test in self.test_tree.iterrows():
                        if is_subtopic(k, test.topic):
                            log.debug(f"test.topic {test.topic} {msg[k]['topic'] + test.topic[len(k):]}")
                            if msg[k]["topic"] == "suggestion":
                                self.test_tree.loc[id, "topic"] = msg[k]["topic"]
                                self.suggestions.loc[id] = self.test_tree.loc[id]
                                self.test_tree.drop(id, inplace=True)
                            else:
                                self.test_tree.loc[id, "topic"] = msg[k]["topic"] + test.topic[len(k):]
                self._update_interface()

            else:
                log.debug(f"Unknown message type! {msg[k]}")

        # except Exception as e:
        #     log.debug(str(traceback.format_exc()))
        #     self._update_interface()
        #     raise e

    def _ensure_add_item_row(self, suggestions):

        # find the outputs in this subtopic
        outputs = []
        comparators = []
        for k, test in self.test_tree.iterrows():
            if is_subtopic(self.current_topic, test.topic):
                outputs.append(test.value2)
                comparators.append(test.comparator)
        outputs = set(outputs)
        common_output = outputs.pop() if len(outputs) == 1 else ""
        try:
            common_comparator = statistics.mode(comparators) if len(comparators) > 0 else "should not be"
        except:
            common_comparator = "should not be"
        # see if we alreay have an add row
        for k, row in suggestions.iterrows():
            if row.value1 == "" and (row.value2 == common_output or row.value2 == ""):
                return suggestions

        # if not we append one
        row = {
            "topic": "suggestion", # will get replaced with imputed version
            "prefix": self.test_tree.iloc[0].prefix if len(self.test_tree) > 0 else "The model output for",
            "value1": "",
            "comparator": common_comparator,
            "value2": common_output,
            "labeler": "imputed"
            # "focus": 0,
            # 'seen': False,
            # 'batch_round': -1,
            # 'label_round': -1,
            # 'focus_topic': "",
            # 'score': np.nan
        }
        for c in self.score_columns:
            row[c] = np.nan
            row[c + " value1 outputs"] = "{}"
            row[c + " value2 outputs"] = "{}"
        new_add = pd.DataFrame([row], index=[uuid.uuid4().hex])

        return suggestions.append(new_add, sort=False)

    def _is_fillin(self, topic=None, include_fillin_root=True):
        if topic is None:
            topic = self.current_topic
        return is_subtopic(FILLIN_PREFIX, topic) if include_fillin_root else topic.startswith(FILLIN_PREFIX + '/')

    def focus_topic_metric(self, k=1, start_from_batch=0):
        """ TODO(Doc)
        For each batch, get examples that were labeled
        only count examples that are in the focus topic or subsets of the focus
        topic that didn't exist so far (assumption: if they did exist, they
        were turned off, otherwise they would be the focus topic)
        """
        tests = self.test_tree
        labeled = tests[tests['label_round'] != -1]
        def batch_scores(batch):
            batch = batch[batch['labeler'] != 'imputed']
            scores = []
            label_start = batch['label_round'].min()
            existing_topics = set(labeled[labeled['label_round'] < label_start]['topic'])
            for _, x in batch[['topic', 'focus_topic', 'score']].iterrows():
                # if x.topic == x.focus_topic or (x.topic.startswith(x.focus_topic) and x.topic not in existing_topics):
                if x.topic == x.focus_topic or (is_subtopic(x.focus_topic, x.topic) and x.topic not in existing_topics):
                    scores.append(x.score)
            return np.array(sorted(scores, reverse=True))
        ret = []
        for r, df in tests.groupby('batch_round'):
            if r == -1:
                continue
            if r < start_from_batch:
                continue
            scores = batch_scores(df)
            if len(scores) < k:
                ret.append(0)
            else:
                ret.append(scores[k-1])
            # print(r, scores)
        return ret

    def _update_interface(self):
        """ Update the interface, 
        but only change the sort order when we are supposed to.
        """
        log.debug("_update_interface()")

        # get the children of the current focus topic
        data = {}
        children = []
        for k, test in self.test_tree.iterrows():
            data[k] = {
                "prefix": test.prefix,
                "value1": test.value1,
                "value1_outputs": {c: [[k, safe_json_load(test.get(c + " value1 outputs", "{}"))]] for c in self.score_columns},
                "comparator": test.comparator,
                "value2": test.value2,
                "value2_outputs": {c: [[k, safe_json_load(test.get(c + " value2 outputs", "{}"))]] for c in self.score_columns},
                "scores": {c: [[k, v] for v in score_parts(test[c])] for c in self.score_columns},
                "topic_name": None,
                "is_topic": False,
                "hidden": self._hidden_topics.get(k, False)
            }

            if is_subtopic(self.current_topic, test.topic):
                if test.topic != self.current_topic:
                    child_topic = test.topic[len(self.current_topic):].split("/")[1]
                    key = self.current_topic+"/"+child_topic
                    if key not in children:
                        # log.debug("key", key, self._hidden_topics.get(key, False))
                        data[key] = {
                            "topic_name": child_topic,
                            "prefix": test.prefix,
                            "scores": {c: [[k, v] for v in score_parts(test[c])] for c in self.score_columns},
                            "value1": None,
                            "value1_outputs": {},
                            "comparator": "should not be",
                            "value2": None,
                            "is_topic": True,
                            "hidden": self._hidden_topics.get(key, False)
                        }
                        if child_topic == "New topic": # we start editing new topics by default
                            data[key]["editing"] = True
                        children.append(key)
                    else:
                        for c in self.score_columns:
                            data[key]["scores"][c].extend([[k, v] for v in score_parts(test[c])])
                else:
                    children.append(k)
        # TODO: This is a complete hack
        children_scores = sorted([np.max([score_max(x[1]) for x in data[key]['scores'][self.score_columns[0]]]) for key in children])
        if len(children_scores) < 5:
            autofilter = -1000000
        else:
            autofilter = min(children_scores[-5] - (children_scores[-1] - children_scores[-5]) * 0.2, 0)


        # log.debug("AUTOFILTER %f" % autofilter)
        # log.debug("in _update_interface2", self.current_topic)
        data["test_chart"] = {
            "suggestions": sorted(list(self.suggestions.index), key=lambda id: -score_max(self.suggestions.loc[id, self.score_columns[0]])),
            "tests": sorted(children, key=lambda id: -max([score_max(s[1]) for s in data[id]["scores"][self.score_columns[0]]])),
            "topic": self.current_topic,
            "score_filter": autofilter if self.score_filter == "auto" else self.score_filter,
            "disable_suggestions": False if self.experiment is None else self.experiment.get("disable_suggestions", False),
            #"test_prefix": self.test_tree.test_prefix,
            "experiment": self.experiment is not None,
            "experiment_locations": None if self.experiment is None else self.experiment.get("locations", None),
            "read_only": self.scorer is None,
            "score_columns": self.score_columns,
            "suggestions_error": self.suggestions_error
        }
        for k, test in self.suggestions.iterrows():
            data[k] = {
                "prefix": test.prefix,
                "value1": test.value1,
                "value1_outputs": {c: [[k, safe_json_load(test.get(c + " value1 outputs", "{}"))]] for c in self.score_columns},
                "comparator": test.comparator,
                "value2": test.value2,
                "value2_outputs": {c: [[k, safe_json_load(test.get(c + " value2 outputs", "{}"))]] for c in self.score_columns},
                "scores": {c: [[k, v] for v in score_parts(test[c])] for c in self.score_columns},
                "is_topic": False,
                "topic_name": None,
                "hidden": self._hidden_topics.get(k, False)
            }
        # log.debug(f"in _update_interface3 {data} x", )
        self.comm.send(data)
    def _finish_updating(self):
        log.warning("before calling alert")
        self.comm.send({'finish_update_alert': None})
        print("after calling alert")

    def _update_lexicons(self):
        if self._checklist_tester is None: # lazy load checklist when needed for baseline user study tests
            self._checklist_tester = checklist.editor.Editor()
            self._checklist_tester.tg
        fillins = collections.defaultdict(lambda:[])
        for _, test in self.test_tree.iterrows():
            if self._is_fillin(test.topic, include_fillin_root=False):
                key = test.topic.split('/')[-1]
                val = test.value1
                fillins[key].append(val)
        for k, vals in fillins.items():
            self._checklist_tester.add_lexicon(k, vals, overwrite=True, remove_duplicates=True)
    
    def _generate_suggestions(self, topic, prompt_threads=None, max_suggestions=None, temperature=None, slot_randomization=None,
                              score_randomization=None, skip_randomization=None, prompt_size=None, complete_diversity=None,
                              prompt_diversity=None, use_focus=None, subtopic_diversity=None, value1_filter=None, comparator_filter=None,
                              value2_filter=None, suggestions_template_value1=None,
                              suggestions_template_comparator=None, suggestions_template_value2=None, checklist_mode=False, generate_outputs=None):
        log.debug(f"_generate_suggestions{topic}")
        # log.debug("suggestions_template", suggestions_template_value1, suggestions_template_comparator, suggestions_template_value2)
        # file_log("suggestions_template", suggestions_template_value1, suggestions_template_comparator, suggestions_template_value2)
        # pull in arg defaults
        if prompt_threads is None:
            prompt_threads = self.prompt_threads
        if max_suggestions is None:
            max_suggestions = self.max_suggestions
        if temperature is None:
            temperature = self.temperature
        if generate_outputs is None:
            generate_outputs = self.generate_outputs

        if checklist_mode:
            if suggestions_template_value1 is None:
                suggestions_template_value1 = ""
            # TODO: fixed comparator here
            self._update_lexicons()
            if self._is_fillin(topic):
                comparator = ''
                value2 = ''
                template = suggestions_template_value1
                if self.experiment is not None and self.experiment.get('model', None) == 'gpt':
                    template = '%s %s' % (suggestions_template_value1, suggestions_template_value2)
                d = [(x, value2) for x in self._checklist_tester.suggest(template, nsamples=max_suggestions)]
            else:
                comparator = 'should not be' if suggestions_template_comparator is None else suggestions_template_comparator
                if self.experiment is not None and self.experiment.get('model', None) == 'gpt':
                    template = '%s %s' % (suggestions_template_value1, suggestions_template_value2)
                    d = [(' '.join(x.split()[:-1]), x.split()[-1]) for x in set(list(self._checklist_tester.template(template, nsamples=max_suggestions, remove_duplicates=True).data))]
                # todo: if gpt, add
                else:
                    d = [(x, suggestions_template_value2) for x in set(list(self._checklist_tester.template(suggestions_template_value1, nsamples=max_suggestions, remove_duplicates=True).data))]
            suggestions = []
            for value1, value2 in d:
                suggestions.append({
                    "topic": "suggestion", # will get replaced with imputed version
                    "prefix": self.test_tree.iloc[0].prefix if len(self.test_tree) > 0 else "The model output for",
                    "value1": value1,
                    "comparator": comparator,
                    "value2": value2,
                    "labeler": "imputed",
                    "focus": 0,
                    'score': np.nan
                    })
            suggestions = pd.DataFrame(suggestions, index=[uuid.uuid4().hex for _ in range(len(suggestions))])
            if not self._is_fillin(topic):
                self._compute_embeddings_and_scores(suggestions)
                suggestions = suggestions.dropna(subset=[self.score_columns[0]])
            suggestions = self._ensure_add_item_row(suggestions)
            return suggestions


        test_map = {}
        for _, test in self.test_tree.iterrows():
            test_map[test["value1"].lower() + " " + test["comparator"].lower() + " " + test["value2"].lower()] = True

        # see if we have a finite set of valid outputs
        valid_outputs = getattr(self.scorer, "output_names", None)
        if valid_outputs is not None and value2_filter is not None:
            valid_outputs = [s for s in valid_outputs if re.search(value2_filter, s) is not None]

        # see if all our outputs seem to be the same
        if valid_outputs is not None and len(valid_outputs) == 1:
            include_value2 = False
        else:
            last_subtopic_output = None
            include_value2 = None
            subtopic_count = 0
            for k, test in self.test_tree.iterrows():
                if is_subtopic(topic, test["topic"]):
                    subtopic_count += 1
                    if last_subtopic_output is None:
                        last_subtopic_output = test["value2"]
                    elif last_subtopic_output != test["value2"]:
                        include_value2 = generate_outputs # if we have more than one output value in our sub-topic children then we can generate outputs
                        break
                ###FK: TODO: I tabbbed this paragraph
            if include_value2 is None:
                if test["comparator"] == "should be the same as for" and subtopic_count < 5:
                    include_value2 = True
                if not include_value2:
                    valid_outputs = [last_subtopic_output]
        log.debug(f"include_value2 = {include_value2}, valid_outputs = {valid_outputs}")
        # we call the model with several prompts to get different threads of ideas
        suggestions = []
        prompts = [self._make_prompt(
                topic,
                slot_randomization=slot_randomization,
                score_randomization=score_randomization,
                skip_randomization=skip_randomization,
                prompt_size=prompt_size,
                complete_diversity=complete_diversity,
                prompt_diversity=prompt_diversity,
                use_focus=use_focus,
                comparator_filter=comparator_filter,
                subtopic_diversity=subtopic_diversity,
                include_value2=include_value2
                ) for _ in range(prompt_threads)]
        #log.debug("prompt", prompts)
        self.backend.temperature = temperature
        proposals = self.backend(prompts, num_samples=max_suggestions // len(prompts))
        # proposals = self._complete_prompt(prompts, n=max_suggestions // prompt_threads, temperature=temperature, valid_outputs=valid_outputs, implicit_output=not include_value2)

        log.debug(f"proposals = {proposals}")

        # try both orderings of all perturbation pairs
        # extended_proposals = []
        # for value1, value2 in proposals:
        #     parts = value1.split(">>>")
        #     if len(parts) == 2:
        #         extended_proposals.append((parts[1]+">>>"+parts[0], value2))
        #     extended_proposals.append((value1, value2))

        # hacky way to get the prefix for the current topic
        prefix = "The model output for"
        for k, test in self.test_tree.iterrows():
            if is_subtopic(topic, test["topic"]):
                prefix = test["prefix"]
                break




        #FK very hacky way for update value 2 according to local model! it depends on task
        map_qqp = {0: 'not duplicate', 1: 'duplicate'}
        map_nli = {0: 'not entailment', 1: 'entailment'}
        map_sentiment = {1: 'positive',  0:'negative'} #TODO 1:'neutral',
        map_toxicity = {1: 'toxic', 0:'non-toxic'} #TODO

        map_review = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}
        # filter out suggestions that are duplicates before we score them
        test_map_tmp = copy.copy(test_map)
        for value1, comparator, value2 in proposals:
            if(self.predictor != None):
                #FK: changing value 2 to prediction
                if(value2 in map_qqp.values()):
                    print(f'in qqp predicting reading {value1} and {comparator} changing {value2} to {map_qqp[np.argmax(self.predictor([value1]))]}')
                    value2 = map_qqp[np.argmax(self.predictor([value1]))]
                elif(value2 in map_sentiment.values()):
                    print(f'in sentiment predicting reading {value1} and {comparator} changing {value2} to {map_sentiment[np.argmax(self.predictor([value1]))]}')
                    value2 = map_sentiment[np.argmax(self.predictor([value1]))]     
                elif(value2 in map_review.values()):
                    print(f'in review  predicting reading {value1} and {comparator} changing {value2} to {map_review[np.argmax(self.predictor([value1]))]}')
                    value2 = map_review[np.argmax(self.predictor([value1]))]     
                elif(value2 in map_nli.values()):
                    print(f'in nli predicting reading {value1} and {comparator} changing {value2} to {map_nli[np.argmax(self.predictor([value1]))]}')
                    value2 = map_nli[np.argmax(self.predictor([value1]))]    
                elif(value2 in map_toxicity.values()):
                    print(f'in nli predicting reading {value1} and {comparator} changing {value2} to {map_toxicity[np.argmax(self.predictor([value1]))]}')
                    value2 = map_toxicity[np.argmax(self.predictor([value1]))]  
                else:
                    print('wrong!!!!!')
            str_val = None
            if value2 is not None:
                str_val = value1.lower() + " " + comparator + " " + value2.lower()
            if str_val not in test_map_tmp:
                s = {
                    "prefix": prefix,
                    "topic": "suggestion", # will get replaced with imputed version
                    "value1": value1,
                    "comparator": comparator,
                    "value2": value2,
                    "labeler": "imputed"
                }
                for c in self.score_columns:
                    s[c] = np.nan
                suggestions.append(s)
                if str_val is not None:
                    test_map_tmp[str_val] = True

        suggestions = pd.DataFrame(suggestions, index=[uuid.uuid4().hex for _ in range(len(suggestions))])
        print('suggestions are', suggestions)
        self._compute_embeddings_and_scores(suggestions)
        suggestions = suggestions.dropna(subset=[self.score_columns[0]])

        # When we have outputs filled in by the scorer we might have more duplicates we need to remove
        duplicates = []
        for k,row in suggestions.iterrows():
            str_val = row.value1.lower() + " " + row.comparator + " " + row.value2.lower()
            if str_val in test_map:
                duplicates.append(k)
            test_map[str_val] = True
        suggestions = suggestions.drop(duplicates)

        if self.topic_model_scale != 0:
            self._add_topic_model_score(suggestions, topic_model_scale=self.topic_model_scale)

        suggestions = self._ensure_add_item_row(suggestions)
        return suggestions

    def _add_topic_model_score(self, df, topic_model_scale):
        documents = []
        for k,s in df.iterrows():
            max_output = -10e8
            max_output_name = None
            for k,v in json.loads(s["score value1 outputs"]).items():
                if v > max_output:
                    max_output = v
                    max_output_name = k
            documents.append(f'"{s["value1"]}" > "{max_output_name}"')

        query = self._make_prompt(
            self.current_topic,
            prompt_size=20,
            include_value2=True
        )["prompt"]

        r = openai.Engine("davinci-instruct-beta").search(
            documents=documents,
            query=query
        )

        sim_scores = np.array([v["score"] for v in r["data"]])
        sim_scores -= np.mean(sim_scores)
        sim_scores /= np.std(sim_scores)

        for i, (k, row) in enumerate(df.iterrows()):
            row["score"] = float(row["score"]) + topic_model_scale * sim_scores[i]

    def _make_prompt(self, topic, prompt_size=None, focus_decay=None, slot_randomization=None, score_randomization=None, skip_randomization=None,
                     use_focus=None, prompt_diversity=None, comparator_filter=None, complete_diversity=None, subtopic_diversity=None, include_value2='auto'):
        """ This builds a prompt for GPT3 that elicits useful input examples.
        """

        log.debug(f"_make_prompt(self, topic={topic}, prompt_size={prompt_size}, focus_decay={focus_decay}, slot_randomization={slot_randomization}, " + \
                  f"score_randomization={score_randomization}, skip_randomization={skip_randomization}, use_focus={use_focus}, prompt_diversity={prompt_diversity}, " \
                  f"complete_diversity={complete_diversity}, subtopic_diversity={subtopic_diversity}, include_value2={include_value2})")

        # pull in arg defaults
        if focus_decay is None:
            focus_decay = self.focus_decay
        if use_focus is None:
            use_focus = self.use_focus
        if prompt_diversity is None:
            prompt_diversity = self.prompt_diversity
        if slot_randomization is None:
            slot_randomization = self.slot_randomization
        if score_randomization is None:
            score_randomization = self.score_randomization
        if skip_randomization is None:
            skip_randomization = self.skip_randomization
        if prompt_size is None:
            prompt_size = self.prompt_size
        if complete_diversity is None:
            complete_diversity = self.complete_diversity
        if subtopic_diversity is None:
            subtopic_diversity = self.subtopic_diversity

        log.debug(f"args after default fill ins: topic={topic}, prompt_size={prompt_size}, focus_decay={focus_decay}, slot_randomization={slot_randomization}, " + \
                  f"score_randomization={score_randomization}, skip_randomization={skip_randomization}, use_focus={use_focus}, prompt_diversity={prompt_diversity}, " \
                  f"complete_diversity={complete_diversity}, subtopic_diversity={subtopic_diversity}, include_value2={include_value2}")

        assert skip_randomization < 0.99, "skip_randomization must be less than 1, otherwise everything will always be skipped!"


        ids = np.array(self.test_tree.index)

        log.debug(f"c ind = {np.where(ids == '25da4ec3a40e419abfb5b7755169c317')}")

        # topic scaling shrinks the priority of IO pairs based on their topic
        # topic_scaling_orig = np.array([1.0 if self.test_tree.loc[k, "topic"].startswith(topic) else 0 for k in ids])
        topic_scaling_orig = []
        for k in ids:
            #FK: made some weird changes here
            if is_subtopic(topic, self.test_tree.loc[k, "topic"]):
                if topic == self.test_tree.loc[k, "topic"]:
                    topic_scaling_orig.append(1.0) # direct children get first priority
                else:
                    topic_scaling_orig.append(0.01) # should be 0.01
            else:
                topic_scaling_orig.append(0.001) #should be 0
        topic_scaling_orig = np.array(topic_scaling_orig)
        # topic_scaling_orig = np.array([1.0 if is_subtopic(topic, self.test_tree.loc[k, "topic"]) else 0 for k in ids])
        topic_scaling = topic_scaling_orig.copy()

        # scale the score of all the items in direct hidden subtopics
        hidden_scaling = np.ones(len(ids))
        for i,k in enumerate(ids):
            sub_topic = self.test_tree.loc[k, "topic"][len(topic)+1:].split("/")[0]
            if self._hidden_topics.get(topic+"/"+sub_topic, False) or self._hidden_topics.get(k, False):
                hidden_scaling[i] = 0.0

            # also hide things if they dont match the filters
            if comparator_filter is not None:
                if re.search(comparator_filter, self.test_tree.loc[k, "comparator"]) is None:
                    hidden_scaling[i] = 0.0

        #compute conflict score
        if(self.check_conflict):
            print("-----------------------------------------computing conlicting score--------------------------------------------")

            engine_name='curie'
            conflict_scores = []
            other_topic_sentences = []
            for k in ids:
                if(self.test_tree.loc[k,'topic']!=topic):
                    sen = self.test_tree.loc[k,'value1']
                    print('non tipic sentence', sen)
                    other_topic_sentences.append(sen)
            prompt = '\n'.join(other_topic_sentences)
            print(topic + '\nthe prompt for other is: ', prompt)
            print('engine is', engine_name)
            embedding_other = get_embedding(prompt, engine=engine_name+'-search-document-msft')

            for k in ids:
                sen = self.test_tree.loc[k,'value1']
                response = get_embedding(sen,engine=engine_name+'-search-query-msft')
                conflict_scores.append(cosine_similarity(embedding_other,response))
                print(sen, cosine_similarity(embedding_other,response) )




        # scores are used directly as the priority for putting something in the prompt
        scores = np.array([score_max(self.test_tree.loc[k, self.score_columns[0]]) for k in ids])
        log.debug(f"np.nanmin(scores) = {np.nanmin(scores)}")
        scores -= np.nanmin(scores) - 1e-8
        scores = np.nan_to_num(scores)

        # randomize the scores a bit to allow for diversity in our prompts
        std_dev = np.sqrt(np.cov(scores, aweights=topic_scaling_orig))
        log.debug(f"score_randomization std = {std_dev}")
        if not np.isnan(std_dev):
            scores += score_randomization * std_dev * np.random.rand(len(ids))

        # avoidance is a vector that marks which items (and items related through similarities) should be avoided (ranked lower for prompt selection)
        if complete_diversity:
            sim_avoidance = np.ones(len(ids))
        elif use_focus:
            sim_avoidance = np.array([self.test_tree.loc[k, "focus"] for k in ids])
        elif prompt_diversity:
            sim_avoidance = np.zeros(len(ids))
        else:
            sim_avoidance = None
        if sim_avoidance is not None:
            embeddings = torch.vstack([torch.tensor(self._embeddings[k]) for k in ids])
            similarities = sentence_transformers.util.pytorch_cos_sim(embeddings, embeddings).numpy()
        hard_avoidance = np.zeros(len(ids))
        diversity = np.ones(len(ids))
        log.debug(f"sim_avoidance is None = {sim_avoidance is None}")

        # compute how many greedy and how many random positions we will have
        num_random = max(0, min(np.random.binomial(prompt_size, slot_randomization), len(ids) - prompt_size))
        num_greedy = max(0, min(prompt_size - num_random, len(ids) - num_random))
        log.debug(f"num_random = {num_random}, num_greedy = {num_greedy}")
        prompt_ids = []
        while len(prompt_ids) < num_greedy + num_random:

            # once we get to the random part of the process we forget what topics we have visited and scramble the scores
            if len(prompt_ids) == num_greedy:
                scores = 1 + np.random.rand(len(ids))*0.1

            # find the next bext index
            if sim_avoidance is not None:
                diversity = 1 - (similarities * sim_avoidance).max(1)
            if(self.check_conflict):
                rank_vals = np.array(scores * topic_scaling * diversity * (1 - hard_avoidance) * hidden_scaling)*np.array(conflict_scores) #np.array(topic_scaling * diversity * (1 - hard_avoidance) * hidden_scaling)*np.array(conflict_scores) #
            else:
                rank_vals = np.array(scores * topic_scaling * diversity * (1 - hard_avoidance) * hidden_scaling)
            log.debug(f"np.nanmax(rank_vals) {np.nanmax(rank_vals)}")

            if np.nanmax(rank_vals) <= 0 and len(prompt_ids) > 0: # stop if we have run out of the current subtree
                break

            new_ind = np.nanargmax(rank_vals)
            log.debug(f"new_ind {new_ind} {self.test_tree.iloc[new_ind]['value1']}")
            if np.random.rand() < skip_randomization:
                log.debug(f"skip {new_ind}")
                avoidance_level = 1 - 0.0001
            else:
                prompt_ids.append(ids[new_ind])
                avoidance_level = 1


            # avoid this IO pair as we select the next pairs
            hard_avoidance[new_ind] = avoidance_level
            if prompt_diversity:
                sim_avoidance[new_ind] = avoidance_level

            # lower the weight of the subtopic we just picked from
            if subtopic_diversity:
                new_topic = self.test_tree.loc[ids[new_ind], "topic"]
                if topic == new_topic:
                    subtopic_scaling = np.ones(len(ids))
                    subtopic_scaling[new_ind] = 0.0001
                else:
                    subtopic = topic + "/" + new_topic[(len(topic)+1):].split("/")[0]
                    # print(subtopic)
                    # subtopic_scaling = np.array([0.0001 if self.test_tree.loc[k, "topic"].startswith(subtopic) else 1 for k in ids])
                    subtopic_scaling = np.array([0.0001 if is_subtopic(subtopic, self.test_tree.loc[k, "topic"]) else 1 for k in ids])
                topic_scaling *= subtopic_scaling

                # topic_scaling = topic_scaling_orig.copy()
                # TODO: should we also turn off avoidance here? (sim_avoidance[:] = 0)

        # update the focus values
        if use_focus:
            for k in ids:
                if topic_scaling_orig[i] == 1:
                    self.test_tree.loc[k, "focus"] *= focus_decay
            for k in prompt_ids:
                self.test_tree.loc[k, "focus"] = min(self.test_tree.loc[k, "focus"] + 0.33, 1)
        log.debug(f"prompt_ids={prompt_ids}, include_value2={include_value2}")

        # see if we are generating comparators because we have multiple types of them
        comparators = set([self.test_tree.loc[k, "comparator"] for k in prompt_ids])
        include_comparator = len(comparators) > 1
        comparator = comparators.pop()

        prompt = []
        for k in reversed(prompt_ids):
            row = self.test_tree.loc[k]
            prompt.append((row["value1"], row["comparator"], row["value2"]))

        return prompt
        # # make the prompt text
        # prompt = ""
        # for k in reversed(prompt_ids):
        #     row = self.test_tree.loc[k]
        #     if include_value2:
        #         if include_comparator:
        #             prompt += f'"{row["value1"]}" {comparator} "{row["value2"]}"\n'
        #         else:
        #             prompt += f'"{row["value1"]}" {self.prompt_seperator} "{row["value2"]}"\n'
        #     else:
        #         prompt += f'"{row["value1"]}"\n'

        # return {
        #     "prompt": prompt + "\"",
        #     "implicit_comparator": None if include_comparator else comparator
        # }

    def _complete_prompt(self, prompts, n, temperature, valid_outputs, implicit_output):
        log.debug(f"_complete_prompt(prompts={prompts}, n={n}, temperature={temperature}")

        response = openai.Completion.create(
            engine=self.engine, prompt=[p["prompt"] for p in prompts], max_tokens=50, # "curie-msft"
            temperature=temperature, n=n, stop="\n"
        )

        lines = [choice["text"] for choice in response["choices"]]
        log.debug(f"response lines = {lines}")

        suggested_tests = []
        for i, line in enumerate(lines):
            match = re.search('^([^"]*)"\W+([^"]*)\W+"([^"]*)"', line)
            if match is not None:
                value1,comparator,value2 = match.groups()
                if prompts[i//n]["implicit_comparator"] is not None:
                    comparator = prompts[i//n]["implicit_comparator"]
                elif comparator not in valid_comparators:
                    comparator = random.choice(valid_comparators)
            elif implicit_output:
                match = re.search('^([^"]*)"[^"]*', line)
                if match is not None:
                    value1 = match.groups()[0]
                    value2 = None if valid_outputs is None or len(valid_outputs) != 1 else valid_outputs[0]
                    comparator = prompts[i//n]["implicit_comparator"]
                else:
                    continue
            else:
                continue

            suggested_tests.append((value1, comparator, value2))

        #log.debug("suggested_tests", suggested_tests)
        return suggested_tests

    def _compute_embeddings_and_scores(self, tests, recompute=False):
        log.debug(f"compute_embeddings_and_scores(tests=<DataFrame shape={tests.shape}>, recompute={recompute})")

        self._compute_scores(tests, recompute=recompute)

        # model outputs and embeddings
        if self.embedding_model is not None:
            new_embedding_ids = [k for k in tests.index if k not in self._embeddings]
            if len(new_embedding_ids) > 0:
                new_value1_embeddings = self.embedding_model.encode([tests.loc[k, "value1"] for k in tests.index], convert_to_tensor=True, show_progress_bar=False).cpu()
                new_value2_embeddings = self.embedding_model.encode([tests.loc[k, "value2"] for k in tests.index], convert_to_tensor=True, show_progress_bar=False).cpu()
                for i,k in enumerate(new_embedding_ids):
                    self._embeddings[k] = np.hstack([new_value1_embeddings[i], new_value2_embeddings[i]])

    def _compute_scores(self, tests, recompute):
        log.debug(f"_compute_scores(tests=<DataFrame shape={tests.shape}>, recompute={recompute})")

        fill_in_scores = np.array([is_subtopic(FILLIN_PREFIX, x) for x in tests['topic']], dtype=bool)
        for i, j in enumerate(np.where(fill_in_scores)[0]):
            tests.loc[tests.index[j], self.score_columns[0]] = np.nan

        if len(self.score_columns) == 0:
            new_score_mask = np.ones(tests.shape[0], dtype=bool)
            new_score_mask[fill_in_scores] = False
        elif recompute:
            new_score_mask = np.ones(len(tests[self.score_columns[0]]), dtype=bool)
            new_score_mask[fill_in_scores] = False
        else:
            new_score_mask = np.array(tests[self.score_columns[0]].isnull())
            new_score_mask[fill_in_scores] = False

        # fill in score_columns if it is empty
        if len(self.score_columns) == 0:
            if isinstance(self.scorer, dict):
                self.score_columns = [k+" score" for k in self.scorer]
            else:
                self.score_columns = ["score"]

            if new_score_mask.sum() == 0:
                for k in self.score_columns:
                    tests[k] = np.zeros(tests.shape[0]) * np.nan

        if new_score_mask.sum() > 0:
            scores = {}
            tests_to_score = tests.loc[new_score_mask, ["value1", "comparator", "value2"]]
            if callable(self.scorer):
                score_vals,scores[self.score_columns[0]+" value1 outputs"],scores[self.score_columns[0]+" value2 outputs"] = self.scorer(tests_to_score)
                scores[self.score_columns[0]] = ["|".join(str(vv) for vv in v) for v in score_vals]
            elif isinstance(self.scorer, dict):
                for i, k in enumerate(self.scorer):
                    score_vals,scores[k+" score value1 outputs"],scores[k+" score value2 outputs"] = self.scorer[k](tests_to_score)
                    scores[k+" score"] = ["|".join(str(vv) for vv in v) for v in score_vals]
            else:
                scores[self.score_columns[0]] = np.zeros(new_score_mask.sum()) * np.nan

            # copy outputs that were generated by the scorer over to the tests dataframe
            for k,v in zip(tests.index, new_score_mask):
                if v:
                    tests.loc[k, "value2"] = tests_to_score.loc[k, "value2"]

            for k in scores:
                for i, j in enumerate(np.where(new_score_mask)[0]):
                    tests.loc[tests.index[j], k] = json.dumps(scores[k][i]) if isinstance(scores[k][i], dict) else scores[k][i]

    def templatize(self, s):
        prompt = """INPUT: "Where are regular people on Twitter"
    OUTPUT: "Where are {regular|normal|sane|typical} people on {Twitter|Facebook|Reddit|Instagram}"
    ###
    INPUT: "Anyone who says this food tastes horrible is out of their mind"
    OUTPUT: "{Anyone|Someone|He|She} who says this food tastes {horrible|terrible|rotten} is out of their mind"
    ###
    INPUT: "great"
    OUTPUT: "{great|excellent|wonderful|superb|delightful}"
    ###
    INPUT: "If you haven't come here before, you probably live under a rock"
    OUTPUT: "If you haven't come here {before|in the past|before now|yet}, you probably {live under a rock|never get out|are a hermit|are isolated}"
    ###
    INPUT: "Only crazy people would say they had a lousy time"
    OUTPUT: "Only {crazy people|insane people|people with no sense|those out of their mind} would say they had a {lousy|terrible|bad|miserable} time"
    ###
    INPUT: "If I didn't come here again, I would be very happy for the rest of my life"
    OUTPUT: "If I didn't come {here|hereabouts|around here} {again|once more|all over again}, I would be very {happy|glad|pleased|elated} for the rest of my life"
    ###
    INPUT: "I don't know what James was talking about when they said they loved the food."
    OUTPUT: "I don't know what {James|John|Robert|Steve|Bob} was talking about when they {said they|stated that|claimed that|mentioned that} they {loved|liked|adored|appreciated} the food."
    ###
    INPUT: "new_input_value"
    OUTPUT: \""""
        prompt = prompt.replace("new_input_value",  s)
        response = openai.Completion.create(
            engine=self.engine, prompt=prompt, max_tokens=300,
            temperature=0.7, n=4, stop="\""
        )

        lines = [choice["text"] for choice in response["choices"]]
        options = []
        for line in lines:
            line = clean_template(line)
            valid = False
            for option in expand_template(line):
                if option == s:
                    valid = True
                    break
            if valid:
                options.append((-len(line), line))
        options.sort()
        log.debug(f"options = {options}")
        return options[0][1]


class TestTree():
    """ A hierarchically organized set of adaptive tests.

    This represents a hierarchically organized set of tests that all target a specific class of models (such as sentiment
    analysis models, or translation models). To interact with a test set you can use either the `__call__` method to
    view and create tests directly in a Jupyter notebook, or you can call the `serve` method to launch a standalone
    webserver. The .tests property of this object is a DataFrame that represents all the tests.
    """

    def __init__(self, tests=None, auto_save=False, verbose=True):#,
                #  valid_output=None, # TODO: delete and replace with scorer util like: `filtered_scorer(scorer, valid_outputs=["good", "better"])`
                #  prompt_seperator=">", test_prefix="The model output for"):
        """ Create a new adaptive test set object.

        Parameters
        ----------
        tests : str or DataFrame or list or None
            The tests to load into this adaptive test set.

        scorer : adatest.Scorer or model
            An adatest.Scorer object that executes a set of tests on a model of interest. If a model function is passed instead
            of an adatest.Scorer object then that model is wrapped as an adatest.Scorer.

        labeler : str
            The name of the user labeling the tests in this adaptive session.
        """


        log.debug(f"__init__(tests={tests}, auto_save={auto_save}, verbose={verbose})")

        # # set the scorer
        # if scorer is None:
        #     scorer = lambda X: np.ones(len(X)) * np.nan
        # # if not isinstance(scorer, Scorer): # auto-wrap any raw model
        # #     scorer = TextScorer(scorer)
        # self.scorer = scorer


        self.verbose = verbose
        # self.prompt_seperator = prompt_seperator
        # self.test_prefix = "" if test_prefix == "" else test_prefix.strip() + " "

        # load the tests
        self._tests_location = tests
        self.auto_save = auto_save
        self._tests = self._load_tests(tests)
        self._last_saved_tests = None#self._tests.copy() #why?

    def _load_tests(self, tests):
        """ Load the given tests into a DataFrame.
        """

        log.debug(f"_load_tests(tests={tests})")

        column_names = ['topic', 'prefix', 'value1', 'comparator', 'value2', 'labeler']
        if tests is None or (isinstance(tests, str) and not os.path.isfile(tests) and self.auto_save):
            return pd.DataFrame([], columns=column_names)

        # load the IO pairs from a csv file if it is given
        if isinstance(tests, str) or isinstance(tests, io.TextIOBase):
            if isinstance(tests, io.TextIOBase) or os.path.isfile(tests):
                tests = pd.read_csv(tests, index_col=0)
                tests["topic"] = tests["topic"].astype(str)
                if "prefix" in tests.columns:
                    tests["prefix"] = tests["prefix"].astype(str)
                tests["value1"] = tests["value1"].astype(str)
                tests["comparator"] = tests["comparator"].astype(str)
                tests["value2"] = tests["value2"].astype(str)
                tests["labeler"] = tests["labeler"].astype(str)

                # for c in tests.columns:
                    # if c.endswith("score"):
                    #     tests[c] = tests[c].astype(float)

                for k, test in tests.iterrows():
                    if test.topic == "nan":
                        tests.loc[k, "topic"] = ""
            else:
                raise Exception(f"The provided tests file does not exist: {tests}")

        # if we are given a list of lists (or tuples) then convert each inner list to a TextIOPair object
        elif isinstance(tests, (list, tuple)) and isinstance(tests[0], (list, tuple)):
            if len(tests[0]) == 3:
                tests = pd.DataFrame(tests, columns=["value1", "comparator", "value2"], index=[uuid.uuid4().hex for _ in range(len(tests))])
            elif len(tests[0]) == 4:
                tests = pd.DataFrame(tests, columns=["prefix", "value1", "comparator", "value2"], index=[uuid.uuid4().hex for _ in range(len(tests))])
            elif len(tests[0]) == 5:
                tests = pd.DataFrame(tests, columns=["topic", "prefix", "value1", "comparator", "value2"], index=[uuid.uuid4().hex for _ in range(len(tests))])
            else:
                raise Exception("When passing list of tuples they need to be of the form (input, output) or (topic, input, output)!")

        assert isinstance(tests, pd.DataFrame), "The passed tests were not in a recognized format!"

        if "topic" not in tests.columns:
            tests["topic"] = ["" for _ in range(tests.shape[0])]

        if "prefix" not in tests.columns:
            tests["prefix"] = ["The model output for" for _ in range(tests.shape[0])]

        # if "score" not in tests.columns:
        #     tests["score"] = [np.nan for _ in range(tests.shape[0])]

        if "labeler" not in tests.columns:
            tests["labeler"] = ["anonymous" for _ in range(tests.shape[0])]

        # if "score" in tests.columns:
        #     tests.rename({"score": "score"}, axis=1, inplace=True)

        # if "focus" not in tests.columns:
        #     tests["focus"] = [0 for _ in range(tests.shape[0])]

        # put the columns in a consistent order
        tests = tests[column_names + [c for c in tests.columns if c not in column_names]]

        if len(set(tests.index)) != len(tests.index):
            raise Exception("The provided tests have duplicate indices!")

        # we ensure all the tests have a score
        # self.compute_scores(tests, recompute=recompute_scores)

        return tests

    # def compute_scores(self, tests, recompute=False):
    #     log.debug(f"compute_scores(tests=<DataFrame shape={tests.shape}>, recompute={recompute})")

    #     fill_in_scores = np.array([is_subtopic(FILLIN_PREFIX, x) for x in tests['topic']], dtype=bool)
    #     for i, j in enumerate(np.where(fill_in_scores)[0]):
    #         tests.loc[tests.index[j], "score"] = np.nan
    #     # scores

    #     if recompute:
    #         new_score_mask = np.ones(len(tests["score"]), dtype=bool)
    #         new_score_mask[fill_in_scores] = False
    #     else:
    #         new_score_mask = tests["score"].isnull()
    #         new_score_mask[fill_in_scores] = False
    #     if new_score_mask.sum() > 0:
    #         score = self.scorer(tests.loc[new_score_mask, ["value1", "comparator", "value2"]])
    #         for i, j in enumerate(np.where(new_score_mask)[0]):
    #             tests.loc[tests.index[j], "score"] = score[i]

    def __getitem__(self, key):
        """ TestSets act just like a DataFrame when sliced.
        """
        return self._tests.__getitem__(key)

    @property
    def loc(self):
        if self.auto_save:
            self._auto_save()
        # loc = self._tests.loc
        # def wrapped_setitem(self, key, value):
        #     if self.auto_save:
        #         self._auto_save()
        #     out = self.__setitem___(key, value)
        #     return out
        # loc.__setitem___ = loc.__setitem__
        # loc.__setitem__ = MethodType(wrapped_setitem, loc)
        return self._tests.loc

    @property
    def iloc(self):
        if self.auto_save:
            self._auto_save()
        return self._tests.iloc

    @property
    def index(self):
        return self._tests.index

    @property
    def columns(self):
        return self._tests.columns

    @property
    def shape(self):
        return self._tests.shape

    @property
    def iterrows(self):
        return self._tests.iterrows

    @property
    def groupby(self):
        return self._tests.groupby

    @property
    def drop(self):
        if self.auto_save:
            self._auto_save()
        return self._tests.drop

    @property
    def insert(self):
        if self.auto_save:
            self._auto_save()
        return self._tests.insert

    def __len__(self):
        return self._tests.__len__()

    def __setitem__(self, key, value):
        return self._tests.__setitem__(key, value)

    def to_csv(self, file):
        self._tests.to_csv(file)

    @debounce(5)
    def _auto_save(self):
        if not self._tests.equals(self._last_saved_tests):
            self.to_csv(self._tests_location)
            self._last_saved_tests = self._tests.copy()
#FK:I changed the default to one & max suggestion to 50
    def __call__(self, scorer=None, starting_topic="", max_suggestions=100, max_suggestions_display=20, prompt_size=7, prompt_threads=10,
                 complete_diversity=False, prompt_diversity=True, use_focus=False, focus_decay=0.8, slot_randomization=0.25,
                 score_randomization=1.0, skip_randomization=0.25, temperature=0.95, subtopic_diversity=True, score_filter="auto",
                 experiment=None, embedding_model=None, user="anonymous", prompt_seperator=">", recompute_scores=False,
                 drop_inactive_scores=True, backend=None, topic_model_scale=0, generate_outputs=True, predictor=None, check_conflict=False, task=None):
        """ Explores the space of input/output pairs in search of problematic examples with high scores.

        scorer : What scorer(s) to use for exploration (if no scorer is given then we are assumed to be in read-only mode).
        """
        log.debug(f"__call__(scorer={scorer}, starting_topic={starting_topic}, max_suggestions={max_suggestions}, " + \
                  f"max_suggestions_display={max_suggestions_display}, prompt_size={prompt_size}, prompt_threads={prompt_threads}, " + \
                  f"complete_diversity={complete_diversity})")

        return TestTreeBrowser(
            self,
            scorer=scorer,
            starting_topic=starting_topic,
            max_suggestions=max_suggestions,
            max_suggestions_display=max_suggestions_display,
            slot_randomization=slot_randomization,
            score_randomization=score_randomization,
            skip_randomization=skip_randomization,
            prompt_size=prompt_size,
            complete_diversity=complete_diversity,
            prompt_diversity=prompt_diversity,
            use_focus=use_focus,
            focus_decay=focus_decay,
            prompt_threads=prompt_threads,
            temperature=temperature,
            subtopic_diversity=subtopic_diversity,
            score_filter=score_filter,
            experiment=experiment,
            embedding_model=embedding_model,
            prompt_seperator=prompt_seperator,
            user=user,
            recompute_scores=recompute_scores,
            backend=backend,
            topic_model_scale=topic_model_scale,
            generate_outputs=generate_outputs,
            drop_inactive_scores=drop_inactive_scores,
            check_conflict=check_conflict,
            predictor = predictor,
            task = task
        )

    def __repr__(self):
        return self._tests.__repr__()

    def _repr_html_(self):
        return self._tests._repr_html_()

def score_max(s):
    if s == "":
        return -10e8
    elif isinstance(s, str):
        return np.max([convert_float(v) for v in s.split("|")])
    elif np.isnan(s):
        return -10e8
    else:
        return np.max(s)

def score_parts(s):
    if isinstance(s, str):
        return [convert_float(v) for v in s.split("|")]
    else:
        return [s]

def convert_float(s):
    try:
        f = float(s)
    except ValueError:
        f = np.nan
    return f

def safe_json_load(input):
    if isinstance(input, float): # catch NaN's
        return {}
    else:
        return json.loads(input)
