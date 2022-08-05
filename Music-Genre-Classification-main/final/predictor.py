import sys
import math
import subprocess
import os
import tensorflow.keras as keras
import numpy as np
import librosa

GENRE_LIST = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ]

def preprocess(file_path, num_mfcc=13, n_fft=2048, hop_length=512):

        samples_per_track = 22050 * 30
        samples_per_segment = int(samples_per_track / 10)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
        mfccs_list = []

        signal, sample_rate = librosa.load(file_path)

        print("Length of signal: {}".format(len(signal)))
        print("Length of samples to consider: {}".format(samples_per_track))

        if len(signal) >= samples_per_track:
            signal = signal[:samples_per_track]

        for x in range(10):
            start = samples_per_segment * x
            finish = start + samples_per_segment
            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                            hop_length=hop_length)
            
            if len(MFCCs) == num_mfcc_vectors_per_segment:
                mfccs_list.append(MFCCs)

        return MFCCs.T



def download_audio_trimmed(youtube_url, start_time, end_time):
        subprocess.call(["youtube-dl", "-x", "--audio-format", "wav", "-o", "input" + ".%(ext)s", youtube_url])
        subprocess.call(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", "input" + ".wav", "-ss", str(start_time), "-t", str(end_time-start_time), "-acodec", "copy", "trimmed_input.wav"])
        os.remove("input.wav")

def download_audio(youtube_url):
    subprocess.call(["youtube-dl", "-x", "--audio-format", "wav", "-o", "input" + ".%(ext)s", youtube_url])
    subprocess.call(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", "input.wav", "-acodec", "copy", "output.wav"])
    os.remove("input.wav")

def main(model_choice="gtzan_model"):
    try:
        model = keras.models.load_model(model_choice)
    except Exception as e:
        sys.exit("Could not load model \"{}\", make sure the model folder is in the same directory that you're running this script in".format(model_choice))

    print("Loaded {}".format(model_choice))

    while True:
        youtube_url = input("Enter Youtube url: ")
        if youtube_url == "quit":
            sys.exit()

        try:
            download_audio(youtube_url)
        except Exception as e:
            print("Couldn't download video. Check the link or try another.")
            continue

        mfccs = preprocess("output.wav")
        mfccs = mfccs[np.newaxis, ..., np.newaxis]

        predictions = model.predict(mfccs)
        predicted_genre = GENRE_LIST[np.argmax(predictions)]

        print("Predicted genre: {}".format(predicted_genre))

        try:
            os.remove("output.wav")
        except Exception as e:
            continue

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit("Please specify \"gtzan_model\" or \"audio_set_model\" as an argument (without quotes)")
    main(sys.argv[1])