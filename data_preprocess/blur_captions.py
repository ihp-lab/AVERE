import os
import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from paddleocr import PaddleOCR

# Set globally per process
ocr = None

def init_worker():
    global ocr
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # Supports Chinese + English

def extract_sampled_frames(video_path, sample_rate=1):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % (fps * sample_rate)) == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def detect_caption_region(frames, bottom_ratio=0.3):
    all_boxes = []
    for frame in frames:
        h, w, _ = frame.shape
        cropped = frame[int(h * (1 - bottom_ratio)):]  # bottom portion
        results = ocr.predict(cropped)
        # print(results)
        # for res in results:
        #     res.save_to_json("output_ocr.json")
        #     print(res["rec_boxes"])
        #     print("====")
        for res in results:
            boxes = res["rec_polys"]
            # print(f"Detected boxes: {boxes}")
            for box in boxes:
                # print(f"Box: {box}")
                box = [[x, y + int(h * (1 - bottom_ratio))] for x, y in box]
                all_boxes.append(box)
    return merge_boxes(all_boxes)

def merge_boxes(boxes, margin=10):
    if not boxes:
        return None
    x_min = min([min([p[0] for p in box]) for box in boxes]) - margin
    y_min = min([min([p[1] for p in box]) for box in boxes]) - margin
    x_max = max([max([p[0] for p in box]) for box in boxes]) + margin
    y_max = max([max([p[1] for p in box]) for box in boxes]) + margin
    return max(0, x_min), max(0, y_min), x_max, y_max

def build_ffmpeg_blur_command(input_path, output_path, blur_box):
    x1, y1, x2, y2 = blur_box
    width, height = x2 - x1, y2 - y1
    filter_str = f"[0:v]crop={width}:{height}:{x1}:{y1},boxblur=20:1[blur];[0:v][blur]overlay={x1}:{y1}[v]"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        str(output_path)
    ]
    return cmd

def process_one_video(args):
    video_path, output_path = args
    try:
        if output_path.exists():
            return f"✅ Skipped (exists): {output_path}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # print(f"🚀 Processing: {video_path}")
        sampled_frames = extract_sampled_frames(video_path, sample_rate=1)
        blur_box = detect_caption_region(sampled_frames)
        if blur_box:
            cmd = build_ffmpeg_blur_command(video_path, output_path, blur_box)
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-c", "copy", str(output_path)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # return f"✅ Done: {video_path}"
    except Exception as e:
        return f"❌ Error processing {video_path}: {e}"

def process_directory_parallel(input_root, output_root, num_workers=4):
    input_root = Path(input_root)
    output_root = Path(output_root)
    all_videos = list(input_root.rglob("*.mp4"))

    tasks = []
    for video_path in all_videos:
        relative_path = video_path.relative_to(input_root)
        output_path = output_root / relative_path
        if not output_path.exists():
            tasks.append((video_path, output_path))

    print(f"🧠 Found {len(tasks)} videos to process across {num_workers} workers.\n")

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for result in tqdm(pool.imap_unordered(process_one_video, tasks), total=len(tasks)):
            # print(result)
            if result is not None:
                print(result)

# === Usage ===
input_root = "/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_video/dfew/dfew_original_clips"
output_root = "/wekafs/ict/achaubey/emotion_reasoning/audio_exp/data_video/dfew/dfew_original_clips_blurred_captions"
process_directory_parallel(input_root, output_root, num_workers=24)
