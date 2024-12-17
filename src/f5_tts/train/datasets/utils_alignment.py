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

# Convert word-level alignment to character-level alignment
def word_to_character_alignment(word_alignments):
    char_alignments = []
    for word_data in word_alignments:
        start_time = word_data['start']
        end_time = word_data['end']
        text = word_data['text']
        
        # Skip invalid or empty words
        if not text or end_time <= start_time:
            continue

        # Calculate the duration per character
        total_duration = end_time - start_time
        char_duration = total_duration / len(text)

        # Assign start and end times for each character
        for i, char in enumerate(text):
            char_start = start_time + i * char_duration
            char_end = char_start + char_duration
            char_alignments.append({
                'char': char,
                'start': round(char_start, 3),
                'end': round(char_end, 3),
            })
    return char_alignments

def word_to_character_alignment(word_alignments, original_text):
    """
    Converts word-level alignments to character-level alignments,
    preserving the original text, including spaces and special characters.

    Args:
        word_alignments (list): A list of dictionaries with word-level alignments.
                                Each dict has 'start', 'end', and 'text' keys.
        original_text (str): The original text (with spaces and formatting).

    Returns:
        list: A list of dictionaries containing character alignments,
              each with 'char', 'start', and 'end' keys.
    """
    char_alignments = []
    current_pos = 0  # Track position in the original text
    for word_data in word_alignments:
        word_text = word_data['text']
        start_time = word_data['start']
        end_time = word_data['end']
        # Skip invalid or empty words
        if not word_text or end_time <= start_time:
            continue
        # Locate the word in the original text, starting from the current position
        word_start_pos = original_text.find(word_text, current_pos)
        if word_start_pos == -1:
            # If the word is not found, skip to avoid misalignment
            continue       
        word_end_pos = word_start_pos + len(word_text)
        total_duration = end_time - start_time
        char_duration = total_duration / len(word_text)
        # Process each character of the word
        for i, char in enumerate(word_text):
            char_start = start_time + i * char_duration
            char_end = char_start + char_duration
            char_alignments.append({
                'char': char,
                'start': round(char_start, 3),
                'end': round(char_end, 3),
            })
        # Update the current position to include the word and its surrounding spaces
        current_pos = word_end_pos
    # Add unaligned spaces and special characters with no duration
    for i, char in enumerate(original_text):
        if not any(char_data['char'] == char and char_data['start'] for char_data in char_alignments):
            char_alignments.append({
                'char': char,
                'start': None,
                'end': None
            })
    return char_alignments

# Create attention matrix
def create_attention_matrix(char_alignments, sample_rate, hop_length):
    # Get the total number of Mel frames
    mel_frames = []
    for char_data in char_alignments:
        char_start_frame = timestamp_to_mel_frame(char_data['start'], sample_rate, hop_length)
        char_end_frame = timestamp_to_mel_frame(char_data['end'], sample_rate, hop_length)
        mel_frames.extend(range(char_start_frame, char_end_frame + 1))
    len_mel = max(mel_frames) + 1  # Total Mel frames
    
    # Initialize attention matrix
    len_text = len(char_alignments)
    attention_matrix = np.zeros((len_text, len_mel), dtype=np.float32)

    # Fill in the attention matrix
    for idx, char_data in enumerate(char_alignments):
        char_start_frame = timestamp_to_mel_frame(char_data['start'], sample_rate, hop_length)
        char_end_frame = timestamp_to_mel_frame(char_data['end'], sample_rate, hop_length)
        attention_matrix[idx, char_start_frame:char_end_frame + 1] = 1.0
    
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
    audio_path = "/workspace/F5-TTS/data/LJSpeech-1.1/wavs/LJ001-0001.wav"
    text = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
    language = "iso"  # ISO-639-3 Language code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16

    print("Len text: ", len(text))
    print("Len mel: ", librosa.get_duration(filename=audio_path) * SAMPLE_RATE // HOP_LENGTH)

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
    attention_matrix = create_attention_matrix(char_alignments, SAMPLE_RATE, HOP_LENGTH)

    # Print Results
    print("\nCharacter Alignments:")
    for char_data in char_alignments:
        print(char_data)

    print("\nAttention Matrix Shape:", attention_matrix.shape)
    print("Attention Matrix:")
    print(attention_matrix)
