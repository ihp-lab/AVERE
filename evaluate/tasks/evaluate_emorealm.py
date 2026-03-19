import glob, os, tqdm, re, json, logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    recall_score,
    confusion_matrix
)
import numpy as np
import pandas as pd
import random
from collections import defaultdict

from utils import chunk_path_list
from eval_constants import EVAL_TEMP_DIR, VIDEO_EVAL_DATA_PAR


short_hand_notation = {
    "visual_reasoning": "VR",
    "audio_reasoning": "AR",
    "modality_agreement": "MA",
    "visual_hallucination": "VH",
    "audio_hallucination": "AH"
}

def rstrip_choice(choice):
    choice = choice.strip()
    if choice.startswith("(A)") or choice.startswith("(B)") or choice.startswith("(C)") or choice.startswith("(D)"):
        choice = choice[3:].strip()
    elif choice.startswith("A:") or choice.startswith("B:") or choice.startswith("C:") or choice.startswith("D:"):
        choice = choice[2:].strip()
    elif choice.startswith("A.") or choice.startswith("B.") or choice.startswith("C.") or choice.startswith("D."):
        choice = choice[2:].strip()
    return choice

def strip_choices_with_letters(choices):
    stripped_choices = []
    for choice in choices:
        new_choice = rstrip_choice(choice)
        stripped_choices.append(new_choice)
    return stripped_choices

def shuffle_choices(choices, correct_choice):
    correct_answer = rstrip_choice(choices[ord(correct_choice.upper()) - ord('A')])
    new_choices = strip_choices_with_letters(choices)
    random.shuffle(new_choices)
    correct_index = new_choices.index(correct_answer)
    new_correct_choice = chr(correct_index + ord('A'))
    new_choices = [f"({chr(i + ord('A'))}) {choice}" for i, choice in enumerate(new_choices)]
    return new_choices, new_correct_choice

def matches_correct_answer(prediction: str, correct_answer: str) -> bool:
    pattern = rf'\b[\(\[]?{re.escape(correct_answer)}[\)\]\.\:]?\b'
    return bool(re.search(pattern, prediction, re.IGNORECASE))

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()

def parse_answer(prediction: str, correct_answer: str, choices: list[str]) -> bool:
    # global RANDOM_COUNT
    correct = (correct_answer or "").upper()
    pred_up = (prediction or "").upper()
    pred_norm = norm(prediction or "")
    # Build {letter -> normalized choice text} and note which letters map to Yes/No
    letter_to_text = {}
    yes_letter = no_letter = None
    for ch in choices or []:
        m = re.match(r'\s*\(?([A-Z])\)?[.)]?\s*(.*)', ch, flags=re.I)
        if not m:
            continue
        L, text = m.group(1).upper(), m.group(2)
        t = norm(text)
        letter_to_text[L] = t
        if 'yes' in t:
            yes_letter = L
        if 'no' in t:
            no_letter = L
    # 1) If prediction contains an explicit letter choice, use it
    for L in letter_to_text.keys():
        if re.search(rf'\b[\(\[]?{L}[\)\]\.\:]?\b', pred_up):
            return L == correct
    # 2) If prediction says Yes/No, map to the corresponding letter from choices
    if re.search(r'\byes\b', pred_norm) and yes_letter:
        return yes_letter == correct
    if re.search(r'\bno\b', pred_norm) and no_letter:
        return no_letter == correct
    # 3) If prediction matches a choice's text (normalized), use that
    for L, t in letter_to_text.items():
        if t and t in pred_norm:
            return L == correct
    # 4) Fallback: random guess among available letters
    # if letter_to_text:
    #     RANDOM_COUNT += 1
    #     return random.choice(list(letter_to_text.keys())) == correct
    # If choices couldn't be parsed, safest is to return False
    return False

def get_metrics_yes_no_qa(items, prediction_save_path, use_parse_answer=False):
    overall_correct = 0
    overall_total = 0
    yes_correct = 0
    yes_total = 0
    no_correct = 0
    no_total = 0
    model_yes_count = 0
    missing_count = 0
    for item in items:
        sample_id = item["id"]
        prediction_file_path = os.path.join(prediction_save_path, f"{sample_id}.txt")
        if not os.path.exists(prediction_file_path):
            missing_count += 1
            continue
        with open(prediction_file_path, "r") as prediction_file:
            prediction = prediction_file.read().strip().lower()
        answer_choice = item["answer"]
        answer_text = rstrip_choice(item["choices"][ord(answer_choice.upper()) - ord('A')]).lower()
        if "yes" in answer_text:
            yes_total += 1
            ans_type = "yes"
        elif "no" in answer_text:
            no_total += 1
            ans_type = "no"
        else:
            raise Exception("Can not determine the answer type -- check!!!")
        # if matches_correct_answer(prediction, answer_choice):
        if use_parse_answer and parse_answer(prediction, answer_choice, item["choices"]):
            cur_correct = True
        elif not use_parse_answer and matches_correct_answer(prediction, answer_choice):
            cur_correct = True
        else:
            cur_correct = False
        if cur_correct:
            overall_correct += 1
            if ans_type == "yes":
                yes_correct += 1
                model_yes_count += 1
            else:
                no_correct += 1
        else:
            if ans_type == "no":
                model_yes_count += 1
        overall_total += 1
    overall_accuracy = overall_correct*100. / overall_total if overall_total > 0 else 0
    precision = yes_correct*100. / (yes_total) if yes_total > 0 else 0
    recall = no_correct*100. / (no_total) if no_total > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    model_yes_percentage = model_yes_count*100. / overall_total if overall_total > 0 else 0
    # print(f"Missing count: {missing_count} out of {len(items)}")
    return {
        "overall_accuracy": overall_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "model_yes_percentage": model_yes_percentage,
    }

def get_metrics_mcq_qa(items, prediction_save_path, use_parse_answer=False):
    overall_correct = 0
    overall_total = 0
    missing_count = 0
    for item in items:
        sample_id = item["id"]
        prediction_file_path = os.path.join(prediction_save_path, f"{sample_id}.txt")
        if not os.path.exists(prediction_file_path):
            missing_count += 1
            continue
        with open(prediction_file_path, "r") as prediction_file:
            prediction = prediction_file.read().strip().lower()
        answer_choice = item["answer"]
        if use_parse_answer:
            if parse_answer(prediction, answer_choice, item["choices"]):
                overall_correct += 1
        else:
            if matches_correct_answer(prediction, answer_choice):
                overall_correct += 1
        overall_total += 1
    overall_accuracy = overall_correct*100. / overall_total if overall_total > 0 else 0
    # print(f"Missing count: {missing_count} out of {len(items)}")
    return {
        "overall_accuracy": overall_accuracy,
    }
       

def get_accuracy(prediction_save_path, benchmark_path, version = "v1", use_parse_answer=False):
    json_path = benchmark_path
    with open(json_path, 'r') as f:
        qa_data = json.load(f)
    qtype_wise_items = defaultdict(list)
    for item in qa_data:
        ## find the category of question
        qtype = item["task"]
        qtype_wise_items[qtype].append(item)
    
    results = {}
    for qtype in qtype_wise_items:
        items = qtype_wise_items[qtype]
        if qtype in ["modality_agreement", "reasoning_stress_audio", "reasoning_stress_video"]:
            metrics = get_metrics_yes_no_qa(items, prediction_save_path, use_parse_answer=use_parse_answer)
        else:
            metrics = get_metrics_mcq_qa(items, prediction_save_path, use_parse_answer=use_parse_answer)
        for k, v in metrics.items():
            results[f"{qtype}_{k}"] = v 
    total_acc = 0
    total_items = 0
    for qtype in qtype_wise_items:
        total_acc += results[f"{qtype}_overall_accuracy"]
        total_items += 1
    results["average_accuracy"] = total_acc / total_items if total_items > 0 else 0
    return results
            

def read_emorealm(par_path, qa_category="all", version="v1"):

    print(f"Evaluating on the following question types: {qa_category}")

    ## read json benchmark file
    json_path = os.path.join(par_path, f"emorealm_{version}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Benchmark file not found at {json_path}")
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    all_fpaths = []
    all_labels = []
    all_sample_ids = []
    all_questions = []
    all_question_types = []

    dfew_par_path = os.path.join(VIDEO_EVAL_DATA_PAR, "dfew")
    

    # print(f"Found {len(data)} items in the EMOREALM dataset")

    for qa in data:
        video_path = os.path.join(dfew_par_path, "dfew_original_clips_24fps", qa["video"])
        # print(f"Video path: {video_path}")
        if not os.path.exists(video_path):
            # print(f"Video file does not exist: {video_path}")
            continue  # Skip if video file does not exist
        audio_path = os.path.join(dfew_par_path, "dfew_original_clips_16khz", qa["video"].replace(".mp4", ".wav"))
        if not os.path.exists(audio_path):
            continue  # Skip if audio file does not exist
        
        if qa["task"] not in qa_category and qa_category != "all":
            continue
        question = qa["question"] + " " + " ".join(qa["choices"])
        answer = qa["answer"]
        sample_id = qa["id"]
        question_type = qa["task"]

        all_fpaths.append({"video": video_path, "audio": audio_path})
        all_labels.append(answer)
        all_sample_ids.append(sample_id)
        all_questions.append(question)
        all_question_types.append(question_type)
    print(f"Found {len(all_fpaths)} items in the EMOREALM dataset")

    return all_fpaths, all_labels, all_sample_ids, all_questions, all_question_types


class EvaluateEMOREALM:

    def __init__(self, with_audio = True, eval_prompt_type = "single_label", qa_category="all", version="v1"):
        self.version = version
        dataset_par_path = os.path.join(VIDEO_EVAL_DATA_PAR, "EmoReAlM")
        self.benchmark_path = os.path.join(dataset_par_path, f"emorealm_{version}.json")
        self.file_paths, self.labels, self.sample_ids, self.prompts, self.question_types = read_emorealm(dataset_par_path, qa_category=qa_category, version=version)

        self.eval_prompt = "custom_qa"


    def evaluate(self, pred_path):
        if self.labels is None:
            raise Exception("The GT answers are None..")
        gts, preds = [], []
        qtypes = []
        for sample_id, label, question_type in zip(self.sample_ids, self.labels, self.question_types):
            save_path = os.path.join(pred_path, f"{sample_id}.txt")
            if not os.path.exists(save_path):
                logging.info(f"Skipped {save_path} as there are no predictions...")
                continue
            with open(save_path) as f:
                pred = f.read().strip().lower()
            qtypes.append(question_type)
            preds.append(pred)
            gts.append(label)
        metrics = get_accuracy(pred_path, self.benchmark_path, version = self.version)
        return metrics