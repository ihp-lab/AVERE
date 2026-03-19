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

from utils import chunk_path_list
from eval_constants import EVAL_TEMP_DIR, VIDEO_EVAL_DATA_PAR


emotion_list = {
    "happiness": ["happiness", "happy", "smiling",  "smile", "pleasant", "content", "contentment", "relaxed", "relaxation", "cheerful", "joy", "amusement", "amused", "positivity", "positive", "lightheartedness", "joyful", "friendly", "warmth", "bright",
                  'crinkle', 'gentle', 'exuberant', 'crow', 'duchenne', 'ecstatic', 'crinkled', 'pleasant', 'contented', 'pleasure', 'cheerful', 'mirth', 'sparkling', 'lightheartedness'],
    "neutral": ["neutral", "neutrality", "calm", "contemplative", "concerned", "thoughtful", "focused", "solemn", "mild", "composed", "serene", "placid", "collected", "stoic", "serious", "attentive", "pensive", "unperturbed", "steady", "meditative", "detached", "reserved", "introspective", "deliberate", "observant", "unmoved", "tranquil",
                'unemotional', 'unexpressive', 'seriousness', 'placidity', 'unreadable', 'normal', 'serious', 'unmoving', 'thoughtful', 'expressionless', 'pensiveness'],
    "surprise": ["surprise", "surprised", "startled", "puzzled", "astonishment", "disbelief", "wonder", "curiosity", "curious", "confusion", "wonder", "bewilderment", "skepticism", "excitement", "uncertainty",
                 'astonishment', 'unexpectedness', 'surprising', 'suddenness', 'startling', 'abruptly', 'amazement', 'expectant', 'astonishing'],
    "fear": ["fear", "shock", "anxiety", "apprehension", "tension", "panic", "alertness", "focus", "worry", "alarm", "curiosity", "contemplation",
             'threat', 'frightening', 'terror', 'panic', 'fright', 'danger', 'vulnerability', 'flight', 'dread', 'darting', 'trembling', 'dilated', 'threatened', 'stimulus', 'gasping', 'quivering'],
    "disgust": ["disgust", "distaste", "disgusted", "discomfort", "displeasure", "displeased", "disapproval", "disdain", "contempt",
                'revulsion', 'distaste', 'aversion', 'rejection', 'repulsive', 'disgusted', 'unpleasant', 'distasteful', 'disapproval', 'smell'],
    "anger": ["anger", "mad", "irate", "outraged", "agitation", "irritated", "enraged", "annoyed", "incensed", "serious", "frustration", "displeased", "stern", "displeasure", "tension",
              'rage', 'simmering', 'irritation', 'fury', 'suppressed', 'fierce', 'resentment', 'shouting', 'snarling', 'annoyance', 'hostile', 'hostility', 'yelling', 'restraint', 'irritated'],
    "sadness": ["sadness", "crying", "distress", "concern", "discomfort", "anguish", "somber", "downcast",
                'melancholy', 'melancholic', 'despair', 'sob', 'weariness', 'weeping', 'quivering', 'hopelessness', 'trembles']
}

def read_mer2023(par_path, with_audio = True):

    # mer2023 annotation
    mer_single_label_df = pd.read_csv(f"{par_path}/test_labels/test3-label.csv")
    classes_map = {"happy":"happiness", "sad":"sadness", "neutral": "neutral", "angry": "anger", "surprise":"surprise"}

    label_dict = {}
    labels_skipped = 0
    for idx, row in mer_single_label_df.iterrows():
        video_id = str(row["name"])
        cur_label = row["discrete"]
        try:
            label_dict[video_id] = classes_map[cur_label]
        except KeyError:
            labels_skipped += 1
            continue

    # find all text files within the dataset
    all_videos = glob.glob(f"{par_path}/test3_24fps/*.mp4")
    all_videos.sort()
    all_labels = []
    all_fpaths = []
    for fpath in all_videos:
        fname = fpath.split("/")[-1].split(".")[0]
        try:
            emotion = label_dict[fname]
        except:
            # skip cases where emotion is not amongst the interested 7.
            continue
        if with_audio:
            audio_fpath = fpath.replace(".mp4", ".wav")
            audio_fpath = audio_fpath.replace("test3_24fps", "test3_16khz")
            if not os.path.exists(audio_fpath):
                logging.warning(f"Audio file {audio_fpath} does not exist, skipping...")
                continue
        all_fpaths.append({"video": fpath, "audio": audio_fpath} if with_audio else {"video": fpath})
        all_labels.append(emotion)
        
    return all_fpaths, all_labels

def remove_neg_sentences(text):
    # define list of negative words and phrases
    negative_words = ['not', 'no', "isn't ", 'nothing', 'cannot', "won't", "shouldn't", "neither", "nor"]
    # segment the text into several sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    # remove sentences with negative words
    positive_sentences = [sentence for sentence in sentences if not any(negative_word in sentence.lower() for negative_word in negative_words)]
    
    return " ".join(positive_sentences)


def description2emotion(description, return_idx = False):
    if description[-1]!=".":
        description+="."
    emotions = ["happiness", "sadness", "surprise", "anger", "neutral"]
    emotions2idx = {emo:i for i, emo in enumerate(emotions)}
    emotions_cnt = {emo:0 for i, emo in enumerate(emotions)}
    
    positive_text = remove_neg_sentences(description)
    if positive_text == '':
        positive_text = description
    
    first_sentence = re.match(r'^(.*?)[.!?]', positive_text).group(1)
    remaining_text = re.sub(r'^(.*?)[.!?]', '', positive_text)
    
    first_sentence_words = re.findall(r'\b\w+\b', first_sentence.lower()) 
    for word in first_sentence_words:
        for emo in emotions:
            if word in emotion_list[emo]:
                if return_idx:
                    return emotions2idx[emo]
                else:
                    return emo
                
    remaining_words = re.findall(r'\b\w+\b', remaining_text.lower()) 
    for word in remaining_words:
        for emo in emotions:
            if word in emotion_list[emo]:
                emotions_cnt[emo] += 1
    max_key = max(emotions_cnt, key=lambda k: emotions_cnt[k])
    if return_idx:
        return emotions2idx[max_key]
    else:
        return max_key

class EvaluateMER2023Test3:

    def __init__(self, with_audio = True, eval_prompt_type = "single_label"):
        dataset_par_path = os.path.join(VIDEO_EVAL_DATA_PAR, "mer2023/MER2023")
        self.file_paths, self.labels = read_mer2023(dataset_par_path, with_audio = with_audio)

        if eval_prompt_type == "single_label":
            self.eval_prompt = "Describe the emotion of the person in the video in one word."
        elif eval_prompt_type == "description":
            self.eval_prompt = "Describe the emotion of the person in the video in detail."
        else:
            raise ValueError(f"Unsupported eval_prompt_type: {eval_prompt_type}. Choose 'single_label' or 'description'.")

    def get_metrics(self, gts, preds):
        """
        Computes accuracy, precision, recall, F1 scores per class, and average recall metrics.

        Args:
            gts (list): List of ground truth class labels.
            preds (list): List of predicted class labels.

        Returns:
            dict: Accuracy, F1, precision, recall, UAR, WAR, and metrics per class.
        """
        preds = np.array(preds)
        gts = np.array(gts)

        # Overall metrics
        overall_accuracy = accuracy_score(gts, preds)
        overall_f1 = f1_score(gts, preds, average="weighted")  # macro = unweighted mean of F1s

        # UAR and WAR
        uar = recall_score(gts, preds, average="macro")      # Unweighted Average Recall
        war = recall_score(gts, preds, average="weighted")   # Weighted Average Recall

        # Per-class metrics
        precision, recall, f1_scores, support = precision_recall_fscore_support(gts, preds, average=None)

        # Class-specific accuracy
        unique_classes = np.unique(np.concatenate([gts, preds]))
        cm = confusion_matrix(gts, preds, labels=unique_classes)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        # Structure the output
        metrics = {
            "overall_accuracy": overall_accuracy,
            "overall_f1_weighted": overall_f1,
            "unweighted_avg_recall": uar,
            "weighted_avg_recall": war,
            "metrics_per_class": {
                cls: {
                    "precision": precision[i],
                    "recall": recall[i],
                    "f1_score": f1_scores[i],
                    "accuracy": class_accuracy[i]
                }
                for i, cls in enumerate(unique_classes)
            }
        }
        return metrics

    def evaluate(self, pred_path):
        if self.labels is None:
            raise Exception("The GT emotions are None..")
        gts, preds = [], []
        for fpath_dict, label in zip(self.file_paths, self.labels):
            fpath = fpath_dict["video"]
            fname = fpath.split("/")[-1].split(".")[0]
            save_path = os.path.join(pred_path, f"{fname}.txt")
            if not os.path.exists(save_path):
                logging.info(f"Skipped {fpath} as there are no predictions...")
                continue
            with open(save_path) as f:
                pred = f.read().strip().lower()
            
            try:
                preds.append(description2emotion(pred))
            except:
                continue
            gts.append(label)
        metrics = self.get_metrics(gts = gts, preds = preds)
        return metrics