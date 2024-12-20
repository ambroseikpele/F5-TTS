import torch
import numpy as np
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
import librosa

# Parameters
SAMPLE_RATE = 24000  # Sampling rate in Hz
HOP_LENGTH = 256     # Hop length for Mel spectrogram computation

# Function to compute Mel frame indices from timestamps
def timestamp_to_mel_frame(timestamp, sample_rate, hop_length):
    return int(timestamp * sample_rate / hop_length)

def word_to_character_alignment(word_alignments, original_text):
    """
    Converts word-level alignments to character-level alignments,
    ensuring all characters, including spaces and punctuation, have valid start and end times.

    Args:
        word_alignments (list): A list of dictionaries with word-level alignments.
                                Each dict has 'start', 'end', and 'text' keys.
        original_text (str): The full original text, including spaces and formatting.

    Returns:
        list: A list of dictionaries with 'char', 'start', and 'end' keys.
    """
    char_alignments = []
    current_pos = 0  # Pointer in the original text
    last_end_time = None  # Tracks the end time of the previous word

    for word_data in word_alignments:
        word_text = word_data['text']
        start_time = word_data['start']
        end_time = word_data['end']

        # Validate input word alignment
        if not word_text or end_time < start_time:
            continue

        # Find the word's position in the original text
        word_start_pos = original_text.find(word_text, current_pos)
        if word_start_pos == -1:
            raise ValueError(f"Word '{word_text}' not found in original_text starting from position {current_pos}")

        # Align spaces or special characters BEFORE this word
        if last_end_time is not None and word_start_pos > current_pos:
            gap_duration = start_time - last_end_time
            gap_char_duration = gap_duration / (word_start_pos - current_pos)

            for i in range(word_start_pos - current_pos):
                char_alignments.append({
                    'char': original_text[current_pos + i],
                    'start': round(last_end_time + i * gap_char_duration, 3),
                    'end': round(last_end_time + (i + 1) * gap_char_duration, 3),
                })
            last_end_time = start_time  # Update the last end time

        # Align characters WITHIN the current word
        word_duration = end_time - start_time
        char_duration = word_duration / len(word_text)

        for i, char in enumerate(word_text):
            char_start = start_time + i * char_duration
            char_end = char_start + char_duration
            char_alignments.append({
                'char': char,
                'start': round(char_start, 3),
                'end': round(char_end, 3),
            })

        # Update tracking variables
        current_pos = word_start_pos + len(word_text)
        last_end_time = end_time

    # Handle any remaining characters AFTER the last word
    if current_pos < len(original_text):
        remaining_duration = 0.1  # Assume a default small duration for trailing characters
        for i in range(current_pos, len(original_text)):
            char_alignments.append({
                'char': original_text[i],
                'start': round(last_end_time, 3),
                'end': round(last_end_time + remaining_duration, 3),
            })
            last_end_time += remaining_duration

    return char_alignments

# Create attention matrix
def create_attention_matrix(char_alignments, sample_rate, hop_length, len_mel):
    len_mel = int(len_mel)
    
    # Initialize attention matrix
    len_text = len(char_alignments)
    attention_matrix = np.zeros((len_text, len_mel), dtype=np.float32)

    # Track the previous token's end frame
    previous_end_frame = 0

    # Fill in the attention matrix
    for idx, char_data in enumerate(char_alignments):
        # Calculate the start and end frames
        char_start_frame = timestamp_to_mel_frame(char_data['start'], sample_rate, hop_length)
        char_end_frame = timestamp_to_mel_frame(char_data['end'], sample_rate, hop_length)
        
        # Prevent overlap by ensuring char_start_frame is at least the previous end frame
        char_start_frame = max(char_start_frame, previous_end_frame)
        char_end_frame = max(char_start_frame, char_end_frame)  # Ensure start <= end

        # Fill the attention matrix
        attention_matrix[idx, char_start_frame:char_end_frame + 1] = 1.0
        
        # Update the previous end frame for the next iteration
        previous_end_frame = char_end_frame + 1

    return attention_matrix

def fix_attention_mask(attn):
    """
    Adjusts the attention mask to ensure no overlap between adjacent tokens.
    Each token's attention span ends before the next token's span begins.

    Args:
        attn (torch.Tensor): Attention mask of shape [b, nt, n], where:
                            b = batch size, nt = number of tokens, n = number of frames.

    Returns:
        torch.Tensor: Adjusted attention mask with no overlaps.
    """
    b, nt, n = attn.shape
    adjusted_attn = torch.zeros_like(attn)  # Initialize a new tensor for adjusted attention masks

    for batch_idx in range(b):
        for token_idx in range(nt):
            # Find the columns (frame indices) where attention is active
            active_frames = (attn[batch_idx, token_idx] > 0).nonzero(as_tuple=True)[0]
            
            if active_frames.numel() == 0:
                continue  # Skip if no active frames for this token

            start = active_frames[0].item()
            end = active_frames[-1].item()

            # Adjust: Make the range exclusive of the last column
            if token_idx < nt - 1:  # Not the last token
                end = min(end, active_frames[-1].item() - 1)

            # Set the adjusted attention range
            adjusted_attn[batch_idx, token_idx, start:end + 1] = 1.0

    return adjusted_attn

# Function to generate word timestamps
def generate_word_timestamps(audio_path, text, alignment_model, alignment_tokenizer, batch_size=16, language='en'):
    """
    Process the audio and text to generate word-level timestamps.

    Args:
        audio_path (str): Path to the audio file.
        text (str): Input text to align.
        alignment_model: Loaded alignment model.
        alignment_tokenizer: Alignment model's tokenizer.
        batch_size (int): Batch size for processing.
        language (str): Language code (ISO-639-3).

    Returns:
        list: Word-level alignments with timestamps.
    """
    # Load audio waveform
    audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)

    # Generate emissions
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )

    # Preprocess text
    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language=language,
    )

    # Get alignments
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    # Get spans
    spans = get_spans(tokens_starred, segments, blank_token)

    # Convert spans to word-level timestamps
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    return word_timestamps

# Main execution block
if __name__ == '__main__':
    # Paths and settings
    audio_path = "data/LJSpeech-1.1/wavs/LJ003-0273.wav"
    text = """"I believe," says Mr. Bennet in the letter already largely quoted,"""
    language = "iso"  # ISO-639-3 Language code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    print("Len text: ", len(text))
    len_mel = librosa.get_duration(filename=audio_path) * SAMPLE_RATE // HOP_LENGTH
    print("Len mel: ", len_mel)

    # Load alignment model
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Step 1: Generate word-level timestamps
    word_timestamps = generate_word_timestamps(
        audio_path, text, alignment_model, alignment_tokenizer, batch_size, language
    )
    print("Word Timestamps:", word_timestamps)

    # Step 2: Convert to character-level alignments
    char_alignments = word_to_character_alignment(word_timestamps, text)

    # Step 3: Create attention matrix
    attention_matrix = create_attention_matrix(char_alignments, SAMPLE_RATE, HOP_LENGTH, len_mel)

    # Print Results
    print("\nCharacter Alignments:")
    for char_data in char_alignments:
        print(char_data)

    print("\nAttention Matrix Shape:", attention_matrix.shape)
    print("Attention Matrix:")
    print(attention_matrix)
    # import pdb; pdb.set_trace()
