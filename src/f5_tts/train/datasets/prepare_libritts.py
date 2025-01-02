import os
import sys

sys.path.append(os.getcwd())

import json
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from datasets.arrow_writer import ArrowWriter
import librosa
from utils_alignment import load_alignment_model, generate_word_timestamps, word_to_character_alignment, create_attention_matrix, convert_word_timestamps_to_phonemes


def deal_with_audio_dir(audio_dir):
    alignment_model, alignment_tokenizer = load_alignment_model(
        "cuda",
        dtype=torch.float16,
    )

    sub_result, durations = [], []
    vocab_set = set()
    audio_lists = list(audio_dir.rglob("*.wav"))

    for line in tqdm(audio_lists):
        text_path = line.with_suffix(".normalized.txt")
        text = open(text_path, "r").read().strip()
        duration = sf.info(line).duration
        if duration < 0.4 or duration > 30:
            continue

        wav_path = str(line)
        try:
            word_timestamps = generate_word_timestamps(wav_path, text, alignment_model, alignment_tokenizer,)
            word_timestamps, phonemized_text = convert_word_timestamps_to_phonemes(word_timestamps)
            char_alignments = word_to_character_alignment(word_timestamps, phonemized_text)
        except:
            continue
        SAMPLE_RATE = 24000
        HOP_LENGTH = 256
        len_mel = librosa.get_duration(path=wav_path) * SAMPLE_RATE // HOP_LENGTH
        attention_matrix = create_attention_matrix(char_alignments, SAMPLE_RATE, HOP_LENGTH, len_mel)
        
        sub_result.append({"audio_path": str(line), "text": phonemized_text, "duration": duration, "attn": attention_matrix.tolist(),})
        durations.append(duration)
        vocab_set.update(list(phonemized_text))
    return sub_result, durations, vocab_set


def main():
    result = []
    duration_list = []
    text_vocab_set = set()

    # process raw data
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = []

    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    CHUNK_SIZE = 2000
    arrow_idx = 0
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
        if len(result) >= CHUNK_SIZE:
            with ArrowWriter(path=f"{save_dir}/raw-{arrow_idx}.arrow") as writer:
                for line in tqdm(result, desc="Writing to raw.arrow ..."):
                    writer.write(line)
            result.clear()
            arrow_idx += 1
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    if result:
        with ArrowWriter(path=f"{save_dir}/raw-{arrow_idx}.arrow") as writer:
            for line in tqdm(result, desc="Writing to raw.arrow ..."):
                writer.write(line)

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list)/3600:.2f} hours")


if __name__ == "__main__":
    max_workers = 4 # 36

    tokenizer = "char"  # "pinyin" | "char"

    SUB_SET = ["train-clean-100", "train-clean-360", "train-other-500"]
    SUB_SET = ["train-clean-360"]
    dataset_dir = "data/LibriTTS_R"
    dataset_name = f"LibriTTS_{'_'.join(SUB_SET)}_{tokenizer}".replace("train-clean-", "").replace("train-other-", "")
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours
