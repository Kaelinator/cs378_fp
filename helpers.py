import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from typing import Tuple
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import datasets
import math

QA_MAX_ANSWER_LENGTH = 30


# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }


# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    # tokenize both questions and the corresponding context
    # if the context length is longer than max_length, we split it to several
    # chunks of max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will help us compute the start_positions
    # and end_positions to get the final answer string.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        # We will label features not containing the answer the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # from the feature idx to sample idx
        sample_index = sample_mapping[i]
        # get the answer for a feature
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and \
                        offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(
                    token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py
def postprocess_qa_predictions(examples,
                               features,
                               predictions: Tuple[np.ndarray, np.ndarray],
                               n_best_size: int = 20):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[
            example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                            -1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[
                          -1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[start_index]) == 0
                            or len(offset_mapping[end_index]) == 0
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or \
                            end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH:
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0],
                                        offset_mapping[end_index][1]),
                            "score": start_logits[start_index] +
                                     end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0})

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions

# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

    def evaluate(self,
                 eval_dataset=None,  # denotes the dataset after mapping
                 eval_examples=None,  # denotes the raw dataset
                 ignore_keys=None,  # keys to be ignored in dataset
                 metric_key_prefix: str = "eval"
                 ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        print('ignore_keys', ignore_keys)
        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits and end_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            
            example_id_to_index = {k: i for i, k in enumerate(eval_examples["id"])}
            # s_ids = [e['id'] for e in sorted(eval_examples, key=lambda t: t[0]['context'].count(" ") + 1)]
            eval_examples_0 = eval_examples.filter(lambda e: 0 < e['context'].count(" ") + 1 < 100)
            eval_examples_100 = eval_examples.filter(lambda e: 100 <= e['context'].count(" ") + 1 < 200)
            eval_examples_200 = eval_examples.filter(lambda e: 200 <= e['context'].count(" ") + 1 < 300)
            eval_examples_300 = eval_examples.filter(lambda e: 300 <= e['context'].count(" ") + 1 < 400)
            eval_examples_400 = eval_examples.filter(lambda e: 400 <= e['context'].count(" ") + 1 < 500)
            eval_examples_500 = eval_examples.filter(lambda e: 500 <= e['context'].count(" ") + 1 < 600)
            eval_examples_rest = eval_examples.filter(lambda e: 600 <= e['context'].count(" ") + 1)

            ids_0 = { e['id'] for e in eval_examples_0 }
            ids_100 = { e['id'] for e in eval_examples_100 }
            ids_200 = { e['id'] for e in eval_examples_200 }
            ids_300 = { e['id'] for e in eval_examples_300 }
            ids_400 = { e['id'] for e in eval_examples_400 }
            ids_500 = { e['id'] for e in eval_examples_500 }
            ids_rest = { e['id'] for e in eval_examples_rest }

            is_0 = { example_id_to_index[e['id']] for e in eval_examples_0 }
            is_100 = { example_id_to_index[e['id']] for e in eval_examples_100 }
            is_200 = { example_id_to_index[e['id']] for e in eval_examples_200 }
            is_300 = { example_id_to_index[e['id']] for e in eval_examples_300 }
            is_400 = { example_id_to_index[e['id']] for e in eval_examples_400 }
            is_500 = { example_id_to_index[e['id']] for e in eval_examples_500 }
            is_rest = { example_id_to_index[e['id']] for e in eval_examples_rest }

            eval_dataset_0 = eval_dataset.filter(lambda e: e['example_id'] in ids_0)
            eval_dataset_100 = eval_dataset.filter(lambda e: e['example_id'] in ids_100)
            eval_dataset_200 = eval_dataset.filter(lambda e: e['example_id'] in ids_200)
            eval_dataset_300 = eval_dataset.filter(lambda e: e['example_id'] in ids_300)
            eval_dataset_400 = eval_dataset.filter(lambda e: e['example_id'] in ids_400)
            eval_dataset_500 = eval_dataset.filter(lambda e: e['example_id'] in ids_500)
            eval_dataset_rest = eval_dataset.filter(lambda e: e['example_id'] in ids_rest)

            predictions_0 = [
              [output.predictions[0][i] for i in range(0, len(output.predictions[0])) if i in is_0],
              [output.predictions[1][i] for i in range(0, len(output.predictions[0])) if i in is_0]
            ]
            predictions_100 = [
              [output.predictions[0][i] for i in range(0, len(output.predictions[0])) if i in is_100],
              [output.predictions[1][i] for i in range(0, len(output.predictions[0])) if i in is_100]
            ]
            predictions_200 = [
              [output.predictions[0][i] for i in range(0, len(output.predictions[0])) if i in is_200],
              [output.predictions[1][i] for i in range(0, len(output.predictions[0])) if i in is_200]
            ]
            predictions_300 = [
              [output.predictions[0][i] for i in range(0, len(output.predictions[0])) if i in is_300],
              [output.predictions[1][i] for i in range(0, len(output.predictions[0])) if i in is_300]
            ]
            predictions_400 = [
              [output.predictions[0][i] for i in range(0, len(output.predictions[0])) if i in is_400],
              [output.predictions[1][i] for i in range(0, len(output.predictions[0])) if i in is_400]
            ]
            predictions_500 = [
              [output.predictions[0][i] for i in range(0, len(output.predictions[0])) if i in is_500],
              [output.predictions[1][i] for i in range(0, len(output.predictions[0])) if i in is_500]
            ]
            predictions_rest = [
              [output.predictions[0][i] for i in range(0, len(output.predictions[0])) if i in is_rest],
              [output.predictions[1][i] for i in range(0, len(output.predictions[0])) if i in is_rest]
            ]

            # post process the raw predictions to get the final prediction
            # (from start_logits, end_logits to an answer string)
            eval_preds_0 = postprocess_qa_predictions(eval_examples_0, eval_dataset_0, predictions_0)
            eval_preds_100 = postprocess_qa_predictions(eval_examples_100, eval_dataset_100, predictions_100)
            eval_preds_200 = postprocess_qa_predictions(eval_examples_200, eval_dataset_200, predictions_200)
            # eval_preds_300 = postprocess_qa_predictions(eval_examples_300, eval_dataset_300, predictions_300)
            eval_preds_400 = postprocess_qa_predictions(eval_examples_400, eval_dataset_400, predictions_400)
            eval_preds_500 = postprocess_qa_predictions(eval_examples_500, eval_dataset_500, predictions_500)
            eval_preds_rest = postprocess_qa_predictions(eval_examples_rest, eval_dataset_rest, predictions_rest)

            formatted_predictions_0 = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds_0.items()]
            references_0 = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples_0]
            formatted_predictions_100 = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds_100.items()]
            references_100 = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples_100]
            formatted_predictions_200 = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds_200.items()]
            references_200 = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples_200]
            formatted_predictions_300 = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds_300.items()]
            references_300 = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples_300]
            formatted_predictions_400 = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds_400.items()]
            references_400 = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples_400]
            formatted_predictions_500 = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds_500.items()]
            references_500 = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples_500]
            formatted_predictions_rest = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds_rest.items()]
            references_rest = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples_rest]

            # compute the metrics according to the predictions and references
            metrics_0 = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions_0,
                               label_ids=references_0)
            )
            metrics_100 = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions_100,
                               label_ids=references_100)
            )
            metrics_200 = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions_200,
                               label_ids=references_200)
            )
            metrics_300 = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions_300,
                               label_ids=references_300)
            )
            metrics_400 = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions_400,
                               label_ids=references_400)
            )
            metrics_500 = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions_500,
                               label_ids=references_500)
            )
            metrics_rest = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions_rest,
                               label_ids=references_rest)
            )


            # # Prefix all keys with metric_key_prefix + '_'
            # for key in list(metrics.keys()):
            #     if not key.startswith(f"{metric_key_prefix}_"):
            #         metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        # self.control = self.callback_handler.on_evaluate(self.args, self.state,
        #                                                  self.control, metrics)
        return {metrics_0, metrics_100, metrics_200,  metrics_300, metrics_400, metrics_500, metrics_rest}

# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
# class QuestionAnsweringTrainer(Trainer):
#     def __init__(self, *args, eval_examples=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.eval_examples = eval_examples
# 
#     def evaluate(self,
#                  eval_dataset=None,  # denotes the dataset after mapping
#                  eval_examples=None,  # denotes the raw dataset
#                  ignore_keys=None,  # keys to be ignored in dataset
#                  metric_key_prefix: str = "eval"
#                  ):
#         eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
#         eval_dataloader = self.get_eval_dataloader(eval_dataset)
#         eval_examples = self.eval_examples if eval_examples is None else eval_examples
# 
#         id_set = { s['id'] for s in eval_examples }
# 
#         # Temporarily disable metric computation, we will do it in the loop here.
#         compute_metrics = self.compute_metrics
#         self.compute_metrics = None
#         try:
#             # compute the raw predictions (start_logits and end_logits)
#             output = self.evaluation_loop(
#                 eval_dataloader,
#                 description="Evaluation",
#                 # No point gathering the predictions if there are no metrics, otherwise we defer to
#                 # self.args.prediction_loss_only
#                 prediction_loss_only=True if compute_metrics is None else None,
#                 ignore_keys=ignore_keys,
#             )
#         finally:
#             self.compute_metrics = compute_metrics
# 
#         if self.compute_metrics is not None:
#             eval_examples = eval_examples.sort('id')
#             eval_dataset = eval_dataset.sort('example_id').filter(lambda e: e['example_id'] in id_set)
#             dataset = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'offset_mapping': [], 'example_id': []}
#             ids = set()
#             for i in range(0, len(eval_examples)):
#                 if eval_dataset[i]['example_id'] in ids:
#                     continue
#                 ids.add(eval_dataset[i]['example_id'])
#                 dataset['input_ids'] += [eval_dataset[i]['input_ids']]
#                 dataset['token_type_ids'] += [eval_dataset[i]['token_type_ids']]
#                 dataset['attention_mask'] += [eval_dataset[i]['attention_mask']]
#                 dataset['offset_mapping'] += [eval_dataset[i]['offset_mapping']]
#                 dataset['example_id'] += [eval_dataset[i]['example_id']]
#             print('LENGTH', len(dataset['example_id']), len(eval_examples))
#             z = zip(eval_examples, datasets.Dataset.from_dict(dataset), output.predictions[0], output.predictions[1])
#             s = sorted(z, key=lambda t: t[0]['context'].count(" ") + 1)
#             
#             total_metrics = []
#             buckets = 7
#             for i in range(0, buckets):
#                 start = math.floor(len(s) / buckets) * i
#                 end = math.floor(len(s) / buckets) * (i + 1)
#                 print('Examples', start, 'to', end)
#                 sorted_examples = { 'id': [], 'title': [], 'context': [], 'question': [], 'answers': [] }
#                 sorted_dataset = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'offset_mapping': [], 'example_id': []}
#                 preds_0 = []
#                 preds_1 = []
#                 skipped = 0
#                 for ex, d, p0, p1 in s[start:end]:
#                     if ex['id'] != d['example_id']:
#                       skipped += 1
#                       continue
#                     sorted_examples['id'] += [ex['id']]
#                     sorted_examples['title'] += [ex['title']]
#                     sorted_examples['context'] += [ex['context']]
#                     sorted_examples['question'] += [ex['question']]
#                     sorted_examples['answers'] += [ex['answers']]
#                     sorted_dataset['input_ids'] += [d['input_ids']]
#                     sorted_dataset['token_type_ids'] += [d['token_type_ids']]
#                     sorted_dataset['attention_mask'] += [d['attention_mask']]
#                     sorted_dataset['offset_mapping'] += [d['offset_mapping']]
#                     sorted_dataset['example_id'] += [d['example_id']]
#                     preds_0 += [p0]
#                     preds_1 += [p1]
#                 print('SKIPPED', skipped)
#                 eval_examples = datasets.Dataset.from_dict(sorted_examples)
#                 eval_dataset = datasets.Dataset.from_dict(sorted_dataset)
#                 preds = [preds_0, preds_1]
#                 # post process the raw predictions to get the final prediction
#                 # (from start_logits, end_logits to an answer string)
#                 eval_preds = postprocess_qa_predictions(eval_examples,
#                                                         eval_dataset,
#                                                         preds)
#                 formatted_predictions = [{"id": k, "prediction_text": v}
#                                          for k, v in eval_preds.items()]
#                 references = [{"id": ex["id"], "answers": ex['answers']}
#                               for ex in eval_examples]
# 
#                 # compute the metrics according to the predictions and references
#                 metrics = self.compute_metrics(
#                     EvalPrediction(predictions=formatted_predictions,
#                                    label_ids=references)
#                 )
# 
#                 # Prefix all keys with metric_key_prefix + '_'
#                 for key in list(metrics.keys()):
#                     if not key.startswith(f"{metric_key_prefix}_"):
#                         metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
# 
#                 print(metrics)
#                 total_metrics += [ metrics ]
#                 self.log(metrics)
#             else:
#                 metrics = {}
# 
#             self.control = self.callback_handler.on_evaluate(self.args, self.state,
#                                                              self.control, metrics)
#         return total_metrics

def count(d, i):
  result = {}
  for e in d:
    c = e[i]
    if c in result:
      result[c] += 1
    else:
      result[c] = 1
  return result

def plot_lengths(dataset):
  # counts = count(dataset.map(lens), 'context_length_word')
  dataset = dataset.map(lens)
  l = [ d['context_length_word'] for d in dataset ]
  fig, ax = plt.subplots()
  ax.hist(l, bins=[0, 100, 200, 300, 400, 500, 600, 700])
  ax.set_xlabel('Word count')
  ax.set_ylabel('Number of examples')
  ax.set_title('Word count distribution for context in SQuAD')
  plt.semilogy(base=10)
  plt.show()

def apply_frequency(dataset, frequency_function):
  d = dataset.to_dict()
  i = 0
  for example in dataset:
    if i % 876 == 0:
      print(i)
    context_word_count = example['context'].count(" ") + 1
    n = frequency_function(context_word_count)
    d['id'] = d['id'] + [example['id'] for i in range(n)]
    d['title'] = d['title'] + [example['title'] for i in range(n)]
    d['context'] = d['context'] + [example['context'] for i in range(n)]
    d['question'] = d['question'] + [example['question'] for i in range(n)]
    d['answers'] = d['answers'] + [example['answers'] for i in range(n)]
    i += 1
  return datasets.Dataset.from_dict(d)

  
def lens(example):
  return {
    'question_length_char': len(example['question']),
    'context_length_char': len(example['context']),
    'answer_lengths_char': list(map(lambda t : len(t), example['answers']['text'])),
    'question_length_word': example['question'].count(" ") + 1,
    'context_length_word': example['context'].count(" ") + 1,
    'answer_lengths_char': list(map(lambda t : t.count(" ") + 1, example['answers']['text'])),
  }

def print_example(example):
  print(example)
  exit()
