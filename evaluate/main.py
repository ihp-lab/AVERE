import os
import argparse
import logging
import glob, os, tqdm, re, json
from tasks import EvaluateDFEW, EvaluateRAVDESSVideo, EvaluateEMOREALM, \
    EvaluateMER2023Test3, EvaluateMER2025OVMERD
from avere_inference import AVEREInference
from eval_constants import EVAL_TEMP_DIR

logging.basicConfig(
    level=logging.INFO,                      # default threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running a benchmark/eval script.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model (or a fine-tuned checkpoint) on a benchmark."
    )

    # --- Required / core paths -------------------------------------------------
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to the model checkpoint or directory you want to evaluate.",
    )

    # --- Optional modifiers ----------------------------------------------------
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
        help="Identifier of the *base* model in case of only projector or lora"
             "Leave empty if model_path already contains all weights.",
    )

    # --- Benchmark selection ---------------------------------------------------

    parser.add_argument(
        "--task",
        type=str,
        choices = ["emotion-dfew-audio", 
                   "emotion-ravdess-video-audio", 
                   "emotion_qa-emorealm", 
                   "emotion-mer2023_test3-audio", 
                   "emotion-mer2025_ovmerd-audio"],
        help="Task for evalutation, e.g. ASR, Emotion Recognition",
    )

    parser.add_argument(
        "--eval_prompt_type",
        type=str,
        default="single_label",
        choices=["single_label", "description"],
        help="Type of evaluation prompt to use for getting prediction from the model. Options are 'single_label' and 'description'. Default is 'single_label'.",
    )

    parser.add_argument(
        "--qa_category",
        type=str,
        default="all",
        help="Category of questions to evaluate on. Default is all categories.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Task for evalutation, e.g. ASR, Emotion Recognition",
    )

    parser.add_argument(
        "--multi_gpu_split",
        action="store_true",
        help="Whether to split the evaluation across multiple GPUs. Only for AudioVideoLLaVA model.",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use if --multi_gpu_split is set. Only for AudioVideoLLaVA model.",
    )

    parser.add_argument(
        "--gpu_split_idx",
        type=int,
        default=0,
        help="Index of the GPU split to use if --multi_gpu_split is set. Only for AudioVideoLLaVA model.",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version of the QA benchmark to use, if applicable.",
    )

    return parser.parse_args()

def _evaluate(args):

    model = AVEREInference(model_path = args.model_path, model_base = args.model_base)

    if args.task.startswith("emotion-ravdess-video"):
        with_audio = True if "audio" in args.task else False
        logging.info(f"Evaluating RAVDESS-Video with audio?: {with_audio}")
        logging.info(f"Using eval prompt type: {args.eval_prompt_type}. Model outputs will vary depending on the prompt type.")
        if "mode" in args.task:
            mode = "audio" if args.task.split("-")[-1] == "audiomode" else "video"
        else:
            mode = "av"
        eval_data = EvaluateRAVDESSVideo(with_audio=with_audio, eval_prompt_type=args.eval_prompt_type, mode=mode)
    elif args.task.startswith("emotion-dfew"):
        with_audio = True if "audio" in args.task else False
        logging.info(f"Evaluating DFEW with audio?: {with_audio}")
        logging.info(f"Using eval prompt type: {args.eval_prompt_type}. Model outputs will vary depending on the prompt type.")
        eval_data = EvaluateDFEW(with_audio=with_audio, eval_prompt_type=args.eval_prompt_type)
    elif args.task.startswith("emotion-mer2025_ovmerd"):
        with_audio = True if "audio" in args.task else False
        logging.info(f"Evaluating MER2025 OV-MERD with audio?: {with_audio}")
        logging.info(f"Using eval prompt type: description. Model outputs will vary depending on the prompt type.")
        eval_data = EvaluateMER2025OVMERD(with_audio=with_audio)
    elif args.task.startswith("emotion_qa-emorealm"):
        with_audio = True
        logging.info(f"Evaluating EMOREALM MCQA Benchmark with audio?: {with_audio}")
        eval_data = EvaluateEMOREALM(with_audio=with_audio, qa_category=args.qa_category, version=args.version)
    else:
        raise NotImplementedError(f"The provided task -- {args.task} -- is not supported yet")
    
    ## create temp path for saving results
    prompt_type_suffix = "_desc" if args.eval_prompt_type == "description" else ""
    version_suffix = f"_{args.version}" if args.task.startswith("emotion_qa-emorealm") else ""
    if re.search(r"/checkpoint-\d+", args.model_path):
        # If the model path contains a checkpoint, we will use the second last part of the path as the model name
        temp_eval_path = os.path.join(EVAL_TEMP_DIR, args.task+prompt_type_suffix+version_suffix, args.model_path.split("/")[-2]+"/"+args.model_path.split("/")[-1])
    else:
        temp_eval_path = os.path.join(EVAL_TEMP_DIR, args.task+prompt_type_suffix+version_suffix, args.model_path.split("/")[-1])
    os.makedirs(temp_eval_path, exist_ok=True)
    if args.qa_category =="all":
        res_path = os.path.join(temp_eval_path, "results.json")
    else:
        res_path = os.path.join(temp_eval_path, f"results_{args.qa_category}.json")
    if os.path.exists(res_path):
        logging.warning(f"Results already exist for the model - {args.model_path} on the dataset - {args.task} at {res_path}")
        metrics = (json.load(open(res_path)))
    else:
        logging.info(f"Starting batch inference...")
        if args.task.split("-")[0].endswith("qa"):
            if getattr(eval_data, "file_paths", None) is not None and getattr(eval_data, "sample_ids", None) is not None:
                all_files = eval_data.file_paths
                all_sample_ids = eval_data.sample_ids
                all_prompts = eval_data.prompts
            else:
                raise ValueError("No file paths or sample ids found in the evaluation data object.")
            pred_path = model.inference_all_qa(eval_prompts=all_prompts, 
                                                all_files=all_files, 
                                                all_sample_ids=all_sample_ids, 
                                                temp_eval_path=temp_eval_path, 
                                                batch_size=args.batch_size,
                                                multi_gpu_split=args.multi_gpu_split,
                                                num_gpus=args.num_gpus,
                                                gpu_split_idx=args.gpu_split_idx)

        else:
            if getattr(eval_data, "audios", None) is not None:
                all_files = eval_data.audios
            elif getattr(eval_data, "file_paths", None) is not None:
                all_files = eval_data.file_paths
            else:
                raise ValueError("No audio or file paths found in the evaluation data object.")
            max_generate_tokens = 1024 if args.task.startswith("emotion-mer2025_ovmerd") else 5
            pred_path = model.inference_all(eval_prompt = eval_data.eval_prompt, 
                                                all_files = all_files, 
                                                temp_eval_path = temp_eval_path, 
                                                batch_size = args.batch_size,
                                                multi_gpu_split=args.multi_gpu_split,
                                                num_gpus=args.num_gpus,
                                                gpu_split_idx=args.gpu_split_idx,
                                                max_new_tokens = max_generate_tokens)
        logging.info(f"Batch inference done... Computing metrics...")
        metrics = eval_data.evaluate(pred_path)
        logging.info(f"Metrics computation done... Saving the results at {res_path}")
        with open(res_path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)
        logging.info(f"Results saved...")

    print(f"=== Results for {args.task} ===")
    print(f"Model Path - {args.model_path}")
    print(f"Model Base - {args.model_base}")
    print(metrics)
    print("================================")



if __name__ == "__main__":
    args = parse_args()
    _evaluate(args)
    