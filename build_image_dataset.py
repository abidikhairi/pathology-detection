import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm


if __name__ == "__main__":
    audio_dir = "data/Audio"

    for subdir in tqdm(os.listdir(audio_dir), desc="Walking through audio directory", leave=False):
        for file in tqdm(os.listdir(os.path.join(audio_dir, subdir)), desc=f"Processing files in {subdir}", leave=False):
            filename = file.replace(".wav", "")
            save_dir = os.path.join("data", "Images", subdir)
            
            os.makedirs(save_dir, exist_ok=True)

            sample_rate, samples = wavfile.read(os.path.join(audio_dir, subdir, file))
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
            
            plt.pcolormesh(times, frequencies, spectrogram)
            plt.imshow(spectrogram)
            plt.savefig(os.path.join(save_dir, filename + ".png"))
