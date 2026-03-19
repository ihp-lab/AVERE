import os
import pandas as pd
import glob
import re
import logging
import json
import time
from openai import OpenAI
import argparse
from datetime import datetime

def read_predictions(prediction_par_dir):
    all_predictions = {}
    for file in os.listdir(prediction_par_dir):
        if file.endswith(".txt"):
            with open(os.path.join(prediction_par_dir, file), 'r') as f:
                all_predictions[file.split(".")[0]] = f.read().strip()
    return all_predictions

def read_ovmerd_gt(ovmerd_path):
    ovmerd_df = pd.read_csv(ovmerd_path)
    gt_descriptions = {}
    for idx, row in ovmerd_df.iterrows():
        video_id = str(row["name"])
        description = row["reason"]
        gt_descriptions[video_id] = description
    return gt_descriptions

def get_evaluation_prompt(gt_description, model_pred):
    evaluation_prompt = """
You will be provided with the ground truth description of a video capturing the emotional quotient of the video.
Your task is to evaluate a given model prediction based by comparing it to the ground truth description.

You need to rate the model prediction on a scale of 1 to 10 on the following criteria:
1. Audio-Visual Cue Overlap: Rate how well the mention of audio-visual events in the prediction aligns with those in the ground truth. A higher score indicates a better match.
2. Emotion-label Consistency: Rate how accurately the predicted emotion from the model aligns with the emotion described in the ground truth. A higher score indicates better consistency.
3. Emotion-cue Association: Only focus on the model response and rate how well the audio-visual cues are associated with the predicted emotion. Rate poorly if an emotion- irrelevant cue is mentioned in the response. A higher score indicates a better association of audio-visual cues with emotion.
4. Hallucinated Cues: Rate the extent to which the model prediction contains hallucinated or fabricated audio-visual cues that are not present in the ground truth. A higher score indicates fewer hallucinations and a lower score indicates more hallucinations.

Return your response in the following json format:
{"cue_overlap": int, "cue_overlap_reason": str, "emotion_consistency": int, "emotion_consistency_reason": str, "emotion_cue_association": int, "emotion_cue_association_reason": str, "hallucinated_cues": int, "hallucinated_cues_reason": str}

Ground Truth Description:"""+gt_description+"""
Model Prediction:"""+model_pred
    return evaluation_prompt

def create_jsonl_gpt(pred_dict, gt_dict, jsonl_path):
    all_requests = []
    missing_responses = 0
    for sample_id, gt_description in gt_dict.items():
        try:
            model_pred = pred_dict[sample_id]
        except:
            missing_responses+=1
            continue
        prompt = get_evaluation_prompt(gt_description, model_pred)
        cur_parts = {"role": "user", "content": prompt}
        cur_request = {"custom_id": f"{sample_id}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [cur_parts],"max_completion_tokens": 400}}
        all_requests.append(cur_request)
    print(f"Total missing responses {missing_responses} out of {len(gt_dict)}")
    with open(jsonl_path, 'w') as f:
        for req in all_requests:
            f.write(json.dumps(req) + '\n')
    print(f"Written {len(all_requests)} requests to {jsonl_path}")

def run_gpt_batch_inference(jsonl_path, output_path, temp_job_dir):
    client = OpenAI()
    batch_input_file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    
    batch_inf = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": jsonl_path
        }
    )

    with open(os.path.join(temp_job_dir, "batch_id.txt"), 'a') as f:
        f.write(f"{datetime.now()} -- {batch_inf.id}\n")

    while batch_inf.status not in ["completed", "failed", "expired", "cancelled"]:
        batch_inf = client.batches.retrieve(batch_inf.id)
        print(f"[{batch_inf.id}] [{jsonl_path}] Batch status: {batch_inf.status}")
        time.sleep(5)



    if batch_inf.status == "completed":
        print("Batch completed successfully!")
        # Retrieve the results
        batch_inf = client.batches.retrieve(batch_inf.id)
        file_response = client.files.content(batch_inf.output_file_id)
        with open(output_path, "w") as f:
            f.write(file_response.text)
        print(f"Written predictions to {output_path}")

def read_predictions_from_jsonl(pred_jsonl_path):
    all_predictions = {}
    missing_responses = 0
    with open(pred_jsonl_path, 'r') as f:
        for line in f:
            resp = json.loads(line.strip())
            sample_id = resp['custom_id']
            try:
                model_response = resp["response"]["body"]["choices"][0]["message"]["content"]
                model_response = model_response.lstrip("```json").rstrip("```").strip()
                model_response = json.loads(model_response)
            except:
                missing_responses+=1
                continue
            all_predictions[sample_id] = model_response
    print(f"Total missing responses {missing_responses} out of {len(all_predictions)+missing_responses}")
    return all_predictions

def get_average_metrics(all_predictions):
    total_cue_overlap = 0
    total_emotion_consistency = 0
    total_emotion_cue_association = 0
    total_hallucinated_cues = 0
    n = len(all_predictions)
    for sample_id, pred in all_predictions.items():
        total_cue_overlap += pred["cue_overlap"]
        total_emotion_consistency += pred["emotion_consistency"]
        total_emotion_cue_association += pred["emotion_cue_association"]
        total_hallucinated_cues += pred["hallucinated_cues"]
    avg_metrics = {
        "avg_cue_overlap": total_cue_overlap/n,
        "avg_emotion_consistency": total_emotion_consistency/n,
        "avg_emotion_cue_association": total_emotion_cue_association/n,
        "avg_hallucinated_cues": total_hallucinated_cues/n
    }
    return avg_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_par_dir", type=str, required=True, help="Directory containing model prediction text files")
    parser.add_argument("--ovmerd_path", type=str, required=False, default="/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_video/mer2025/track3_train_ovmerd.csv", help="Path to the OV-MERD CSV file")
    parser.add_argument("--temp_job_dir", type=str, required=True, help="Temporary directory for job files")
    args = parser.parse_args()

    os.makedirs(args.temp_job_dir, exist_ok=True)
    requests_jsonl_path = os.path.join(args.temp_job_dir, "gpt_evaluation_requests.jsonl")
    output_jsonl_path = os.path.join(args.temp_job_dir, "gpt_evaluation_responses.jsonl")

    pred_dict = read_predictions(args.prediction_par_dir)
    gt_dict = read_ovmerd_gt(args.ovmerd_path)
    create_jsonl_gpt(pred_dict, gt_dict, requests_jsonl_path)
    if os.path.exists(output_jsonl_path):
        print(f"Output file {output_jsonl_path} already exists. Skipping batch inference.")
    else:
        run_gpt_batch_inference(requests_jsonl_path, output_jsonl_path, args.temp_job_dir)
    all_predictions = read_predictions_from_jsonl(output_jsonl_path)
    avg_metrics = get_average_metrics(all_predictions)
    with open(os.path.join(args.temp_job_dir, "average_metrics.json"), 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    print("Average Metrics:", avg_metrics)

if __name__ == "__main__":
    main()
    