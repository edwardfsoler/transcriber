# Use a pipeline as a high-level helper
from transformers import pipeline
import torchaudio
import torch
import time
from typing import Optional
import subprocess

def transcribe_audio_file(file_path: str, model_size: str = "large-v2", show_prediction: bool = False,
                          save_file_path: Optional[str] = None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-" + model_size,
        chunk_length_s=30,
        device=device,
    )

    if file_path.endswith(".m4a"):
        print("Converting m4a to wav file")
        cmd = ["ffmpeg",
              "-i", file_path,
              "-acodec", "pcm_s16le",
              "-ar", "44100",
              file_path.replace(".m4a", ".wav")
              ]
        subprocess.run(cmd)
        file_path = file_path.replace(".m4a", ".wav")

    wave_form, sample_rate = torchaudio.load(file_path)
    wave_form_input = wave_form.numpy()[0]

    print("Starting inference...")

    if device == "cpu":
        print("Running inference on CPU")
    else:
        print("Running inference on GPU")

    # we can also return timestamps for the predictions
    start = time.time()
    prediction = pipe(wave_form_input.copy(), batch_size=8)["text"]
    print("Inference completed in %.2f seconds" % (time.time() - start))

    if show_prediction:
        print("Prediction:")
        print(prediction)

    if save_file_path:
        with open(save_file_path, "w") as f:
            f.write(prediction)
        print("Transcription saved at %s" % save_file_path)


if __name__ == "__main__":
    transcribe_audio_file("/home/edwardfsoler/Downloads/Sanding 2.m4a",
                          #save_file_path="/home/edwardfsoler/Desktop/Dru 360 Interview Transcribed.txt",
                          show_prediction=True)
