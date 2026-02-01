# Samson/src/audio_processing_suite/audio_processing.py
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T # For T.Resample as used in the new function
import soundfile as sf
import numpy as np
import re # For safe filenames
from collections import defaultdict # For the new function
from typing import Dict, Tuple, Optional, List, Any # Updated typing imports
import io # <<< ADDED for in-memory byte streams

from src.logger_setup import logger

# This module provides generic audio processing utilities that are designed to be
# model-agnostic. Callers are responsible for handling model-specific requirements
# such as target sample rates or input tensor types (e.g., converting torch.Tensor
# to numpy.ndarray or mlx.core.array if needed by a specific model like Parakeet MLX).

def load_audio(audio_path: Path, target_sr: int = 16000) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Loads an audio file, resamples it to a target sample rate, and converts it to a mono Tensor.

    Args:
        audio_path: Path to the audio file.
        target_sr: The desired sample rate for the output waveform. Callers should set this
                   based on the requirements of the specific STT model being used (e.g., Whisper, Parakeet).

    Returns:
        A tuple containing the waveform (torch.Tensor) and its sample rate (int),
        or (None, None) if loading fails. The waveform is typically float32 and in the range [-1.0, 1.0].
    """
    logger.info(f"Loading audio from: {audio_path}")
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path)) # Convert Path to str for torchaudio

        # Resample if necessary
        if sample_rate != target_sr:
            logger.info(f"Resampling audio from {sample_rate} Hz to {target_sr} Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr

        # Convert to mono if necessary (average channels)
        if waveform.shape[0] > 1:
            logger.info(f"Converting audio from {waveform.shape[0]} channels to mono.")
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure the waveform is float32, common expectation for models
        if waveform.dtype != torch.float32:
            logger.info(f"Converting waveform from {waveform.dtype} to torch.float32.")
            waveform = waveform.float()
            
        logger.info(f"Audio loaded. Shape: {waveform.shape}, Sample Rate: {sample_rate} Hz, Dtype: {waveform.dtype}")
        return waveform, sample_rate

    except FileNotFoundError:
        logger.error(f"Audio file not found at {audio_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}", exc_info=True)
        return None, None

# Helper function for saving audio, as used by the new extract_speaker_audio_segments
def save_audio(audio_tensor: torch.Tensor, path: Path, sample_rate: int):
    """Saves an audio tensor to a WAV file using soundfile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare audio_data for soundfile.write
    # It expects data as NumPy array: (frames,) for mono, (frames, channels) for stereo
    if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 1: # Mono [1, samples]
        audio_data = audio_tensor.squeeze(0).cpu().numpy()
    elif audio_tensor.ndim == 1: # Mono [samples]
        audio_data = audio_tensor.cpu().numpy()
    elif audio_tensor.ndim == 2 and audio_tensor.shape[0] > 1 : # Multi-channel [channels, samples]
        # Transpose to [samples, channels] for soundfile
        audio_data = audio_tensor.cpu().numpy().T 
    else:
        logger.error(f"Unsupported audio_tensor shape for saving: {audio_tensor.shape}. Path: {path}")
        return

    # Ensure float32, then sf.write will handle conversion to PCM_16 if specified
    if audio_data.dtype != np.float32:
         audio_data = audio_data.astype(np.float32)
             
    try:
        sf.write(str(path), audio_data, sample_rate, subtype='PCM_16')
        logger.debug(f"Successfully saved audio to {path}")
    except Exception as e:
        logger.warning(f"Failed to save audio to {path}: {e}", exc_info=True) # Use warning to match user's preference in new code

# Updated extract_speaker_audio_segments function
def extract_speaker_audio_segments(
    full_audio_tensor: torch.Tensor,
    sample_rate: int,
    word_level_transcript: List[Dict[str, Any]], # This is ASR output with word-level speaker tags
    min_duration_s: float = 2.0,
    target_sr: int = 16000,
    output_dir: Optional[Path] = None # For saving snippets
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[Dict[str, Any]]]]:
    logger.info(f"Starting extraction of speaker audio segments (Min Duration: {min_duration_s}s, Target SR: {target_sr}Hz).")
    speaker_audio_segments: Dict[str, List[torch.Tensor]] = defaultdict(list)
    speaker_word_segments_info: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if full_audio_tensor is None:
        logger.error("Full audio tensor is None, cannot extract speaker segments.")
        return {}, {}
    if full_audio_tensor.numel() == 0: # Check if tensor has any elements
        logger.warning("Full audio tensor is empty, cannot extract speaker segments.")
        return {}, {}
    # Ensure full_audio_tensor is at least 2D [channels, samples] for slicing waveform[:, start:end]
    if full_audio_tensor.ndim == 1:
        full_audio_tensor = full_audio_tensor.unsqueeze(0)


    if not word_level_transcript:
        logger.warning("Word level transcript is empty, cannot extract speaker segments.")
        return {}, {}

    resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr) if sample_rate != target_sr else None

    logger.info("Extracting audio chunks by grouping consecutive words from the same speaker...")
    
    all_words_with_speaker: List[Dict[str, Any]] = []
    for segment in word_level_transcript:
        if "words" in segment and isinstance(segment["words"], list):
            all_words_with_speaker.extend(segment["words"])
        # Handle cases where top-level items in word_level_transcript are words themselves (e.g. from whisper word_timestamps=True)
        elif all(key in segment for key in ["word", "start", "end", "speaker"]): 
            all_words_with_speaker.append(segment)


    if not all_words_with_speaker:
        logger.warning("No words found in transcript to process for speaker audio extraction.")
        return {}, {}
        
    # Sort all words by start time just in case segments were out of order
    all_words_with_speaker.sort(key=lambda x: x.get('start', float('inf')))

    current_speaker_label: Optional[str] = None
    current_speaker_chunk_words: List[Dict[str, Any]] = []
    
    for word_info in all_words_with_speaker:
        speaker = str(word_info.get("speaker", "UNKNOWN_SPEAKER"))
        word_text = str(word_info.get("word", "")).strip() # Ensure word is string and stripped
        start_time = word_info.get("start")
        end_time = word_info.get("end")

        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            logger.debug(f"Skipping word '{word_text}' due to invalid start/end times: {start_time}, {end_time}")
            continue
        
        if not word_text: # Allow words with text but no explicit timing if logic changes, but for now timing is crucial
             logger.debug(f"Skipping word due to empty text at {start_time}-{end_time} by {speaker}")
             continue


        if speaker != current_speaker_label:
            # Process previous speaker's chunk if it meets criteria
            if current_speaker_label and current_speaker_chunk_words:
                chunk_start_time = current_speaker_chunk_words[0]['start']
                chunk_end_time = current_speaker_chunk_words[-1]['end']
                duration = chunk_end_time - chunk_start_time
                
                if duration >= min_duration_s:
                    start_sample = int(chunk_start_time * sample_rate)
                    end_sample = int(chunk_end_time * sample_rate)
                    
                    # Boundary check for audio tensor
                    if start_sample < 0: start_sample = 0
                    if end_sample > full_audio_tensor.shape[1]: end_sample = full_audio_tensor.shape[1]

                    if start_sample < end_sample : # Ensure there's actually audio to slice
                        audio_chunk = full_audio_tensor[:, start_sample:end_sample]
                        
                        if resampler:
                            audio_chunk = resampler(audio_chunk)
                        
                        speaker_audio_segments[current_speaker_label].append(audio_chunk)
                        speaker_word_segments_info[current_speaker_label].append({
                            "start": chunk_start_time, "end": chunk_end_time, "duration": duration,
                            "num_words": len(current_speaker_chunk_words)
                            # "words": [w['word'] for w in current_speaker_chunk_words] # Optional: for debugging
                        })
                        if output_dir and current_speaker_label not in ["UNKNOWN_SPEAKER"]:
                            safe_speaker_label = re.sub(r'[^\w\-_.]', '_', current_speaker_label) # Allow period
                            # Ensure unique filenames if multiple chunks from same speaker have similar start/end due to rounding
                            num_existing_segments = len(speaker_audio_segments[current_speaker_label])
                            snippet_filename = output_dir / f"{safe_speaker_label}_chunk_{num_existing_segments}_{chunk_start_time:.2f}_{chunk_end_time:.2f}.wav"
                            try:
                                save_audio(audio_chunk, snippet_filename, target_sr) # Save at target_sr
                                # logger.debug(f"Saved audio snippet: {snippet_filename}") # save_audio logs this
                            except Exception as e_save:
                                logger.warning(f"Could not save audio snippet {snippet_filename}: {e_save}")
                    else:
                        logger.debug(f"Skipping empty audio slice for speaker {current_speaker_label} from {chunk_start_time:.2f}s to {chunk_end_time:.2f}s after boundary checks.")

            # Reset for new speaker
            current_speaker_label = speaker
            current_speaker_chunk_words = [word_info]
        else:
            # Continue current speaker's chunk
            current_speaker_chunk_words.append(word_info)

    # Process the very last chunk after loop finishes
    if current_speaker_label and current_speaker_chunk_words:
        chunk_start_time = current_speaker_chunk_words[0]['start']
        chunk_end_time = current_speaker_chunk_words[-1]['end']
        duration = chunk_end_time - chunk_start_time
        
        if duration >= min_duration_s:
            start_sample = int(chunk_start_time * sample_rate)
            end_sample = int(chunk_end_time * sample_rate)

            if start_sample < 0: start_sample = 0
            if end_sample > full_audio_tensor.shape[1]: end_sample = full_audio_tensor.shape[1]

            if start_sample < end_sample:
                audio_chunk = full_audio_tensor[:, start_sample:end_sample]

                if resampler:
                    audio_chunk = resampler(audio_chunk)
                    
                speaker_audio_segments[current_speaker_label].append(audio_chunk)
                speaker_word_segments_info[current_speaker_label].append({
                    "start": chunk_start_time, "end": chunk_end_time, "duration": duration,
                    "num_words": len(current_speaker_chunk_words)
                    # "words": [w['word'] for w in current_speaker_chunk_words] # Optional: for debugging
                })
                if output_dir and current_speaker_label not in ["UNKNOWN_SPEAKER"]:
                    safe_speaker_label = re.sub(r'[^\w\-_.]', '_', current_speaker_label) # Allow period
                    num_existing_segments = len(speaker_audio_segments[current_speaker_label])
                    snippet_filename = output_dir / f"{safe_speaker_label}_chunk_{num_existing_segments}_{chunk_start_time:.2f}_{chunk_end_time:.2f}.wav"
                    try:
                        save_audio(audio_chunk, snippet_filename, target_sr) # Save at target_sr
                        # logger.debug(f"Saved audio snippet: {snippet_filename}") # save_audio logs this
                    except Exception as e_save:
                        logger.warning(f"Could not save audio snippet {snippet_filename}: {e_save}")
            else:
                logger.debug(f"Skipping empty audio slice for final chunk for speaker {current_speaker_label} from {chunk_start_time:.2f}s to {chunk_end_time:.2f}s after boundary checks.")


    num_extracted_segments = sum(len(v) for v in speaker_audio_segments.values())
    logger.info(f"Audio segment extraction complete. Extracted {num_extracted_segments} segments for {len(speaker_audio_segments)} unique speaker labels meeting min duration.")
    for spk, infos in speaker_word_segments_info.items():
        if infos: # Only log if there are segments for this speaker
            durations_str = ", ".join([f"{info['duration']:.2f}s" for info in infos])
            logger.debug(f"  Speaker {spk}: {len(infos)} segments. Durations: [{durations_str}]")
        
    return speaker_audio_segments, speaker_word_segments_info


def save_speaker_snippet(
    full_waveform: torch.Tensor,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    max_snippet_sec: float,
    output_dir: Path, 
    filename_base: str
    ) -> Optional[str]:
    """
    Extracts a short audio snippet for a speaker and saves it as a WAV file.
    (This function might be deprecated or used for other purposes if the new extract_speaker_audio_segments
     handles all required snippet saving for speaker ID.)
    """
    if full_waveform is None or sample_rate is None:
        logger.error("Cannot save snippet: Waveform or sample rate is missing.")
        return None
    
    # Ensure full_waveform is at least 2D [channels, samples]
    if full_waveform.ndim == 1:
        full_waveform = full_waveform.unsqueeze(0)


    try:
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)

        # Calculate snippet duration and adjust end sample if needed
        max_snippet_samples = int(max_snippet_sec * sample_rate)
        snippet_end_sample = min(end_sample, start_sample + max_snippet_samples)

        # Ensure indices are valid
        if start_sample < 0: start_sample = 0
        if snippet_end_sample > full_waveform.shape[1]: snippet_end_sample = full_waveform.shape[1]
        
        if start_sample >= snippet_end_sample:
            logger.warning(f"Invalid sample range for snippet: {start_sample}-{snippet_end_sample} (Waveform shape: {full_waveform.shape}). Cannot save snippet for {filename_base}.")
            return None

        # Extract snippet tensor
        snippet_tensor = full_waveform[:, start_sample:snippet_end_sample]

        # Prepare filename and path
        snippet_filename = f"{filename_base}.wav" 
        output_path: Path = output_dir / snippet_filename

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save using the new save_audio helper for consistency, or keep sf.write directly
        save_audio(snippet_tensor, output_path, sample_rate) # Assumes snippet is not resampled by this func

        # logger.info(f"Saved speaker snippet to {output_path}") # save_audio will log
        return str(output_path)

    except Exception as e:
        logger.error(f"Error saving snippet for {filename_base} to {output_dir}: {e}", exc_info=True)
        return None

# <<< START OF NEW FUNCTION >>>
def extract_audio_snippet(
    full_waveform: torch.Tensor,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> Optional[bytes]:
    """
    Extracts a short audio snippet and returns it as WAV-formatted bytes.

    Args:
        full_waveform: The complete audio waveform as a PyTorch tensor.
        sample_rate: The sample rate of the waveform.
        start_sec: The start time of the snippet in seconds.
        end_sec: The end time of the snippet in seconds.

    Returns:
        A bytes object containing the WAV-encoded audio snippet, or None on failure.
    """
    if full_waveform is None or sample_rate is None:
        logger.error("Cannot extract snippet: Waveform or sample rate is missing.")
        return None

    if full_waveform.ndim == 1:
        full_waveform = full_waveform.unsqueeze(0)

    try:
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)

        # Ensure indices are valid and within the bounds of the tensor
        start_sample = max(0, start_sample)
        end_sample = min(full_waveform.shape[1], end_sample)

        if start_sample >= end_sample:
            logger.warning(f"Invalid sample range for snippet extraction: {start_sample}-{end_sample}. Waveform shape: {full_waveform.shape}.")
            return None

        # Extract the snippet tensor
        snippet_tensor = full_waveform[:, start_sample:end_sample]

        # Use an in-memory buffer to save the snippet as WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, snippet_tensor.cpu(), sample_rate, format="wav")
        buffer.seek(0)
        
        wav_bytes = buffer.read()
        logger.debug(f"Successfully extracted audio snippet ({start_sec:.2f}s - {end_sec:.2f}s) into {len(wav_bytes)} bytes.")
        return wav_bytes

    except Exception as e:
        logger.error(f"Error extracting audio snippet bytes from {start_sec:.2f}s to {end_sec:.2f}s: {e}", exc_info=True)
        return None
# <<< END OF NEW FUNCTION >>>


def has_speech(
    wav_file_path: Path,
    energy_threshold: float = 0.001,
    speech_percentage_threshold: float = 1.0
) -> bool:
    """
    Performs simple Voice Activity Detection (VAD) based on audio energy.

    Args:
        wav_file_path: Path to the input WAV audio file.
        energy_threshold: The normalized RMS energy level above which a frame is considered speech.
        speech_percentage_threshold: The minimum percentage of frames that must be
                                     considered speech for the file to pass the check.

    Returns:
        True if the amount of speech detected is above the threshold, False otherwise.
    """
    try:
        waveform, sample_rate = torchaudio.load(wav_file_path)

        # Ensure mono audio for consistent RMS calculation
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        frame_size = int(0.02 * sample_rate)  # 20ms frames
        hop_size = int(0.01 * sample_rate)    # 10ms hop

        if waveform.shape[1] < frame_size:
            logger.debug(f"VAD: Waveform for '{wav_file_path.name}' is too short for framing. Assuming no speech.")
            return False

        frames = waveform.unfold(1, frame_size, hop_size)
        
        # Calculate Root Mean Square (RMS) energy for each frame
        rms = torch.sqrt(torch.mean(torch.square(frames), dim=2)).squeeze()
        
        # Normalize RMS to range [0, 1]
        max_rms = torch.max(rms)
        if max_rms > 0:
            rms = rms / max_rms

        num_speech_frames = torch.sum(rms > energy_threshold).item()
        total_frames = rms.shape[0]

        if total_frames == 0:
            return False

        speech_percentage = (num_speech_frames / total_frames) * 100
        
        logger.debug(
            f"VAD check for '{wav_file_path.name}': "
            f"{num_speech_frames}/{total_frames} frames ({speech_percentage:.1f}%) "
            f"exceeded energy threshold {energy_threshold}."
        )

        return speech_percentage > speech_percentage_threshold

    except Exception as e:
        logger.error(f"VAD check failed for {wav_file_path.name}: {e}", exc_info=True)
        # Fail safe: assume there is speech if the VAD process itself fails.
        return True