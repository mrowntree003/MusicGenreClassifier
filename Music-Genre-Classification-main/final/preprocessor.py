import os
import librosa
import math
import json

MARSYAS_DATASET_PATH = "../marsyas_data"
MARSYAS_JSON_PATH = "../marsyas_data.json"

AUDIO_SET_DATASET_PATH = "../audio_set_data"
AUDIO_SET_JSON_PATH = "../audio_set_data.json"

SAMPLE_RATE = 22050
MARSYAS_DURATION = 30
AUDIO_SET_DURATION = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * MARSYAS_DURATION # make sure to change this to "SAMPLE_RATE * AUDIO_SET_DURATION" if you're using audio set


def main():
    dataset_path = MARSYAS_DATASET_PATH # change depending on which data set you're processing
    json_path = MARSYAS_JSON_PATH # change depending on which data set you're processing
    n_mfcc=13
    n_fft=2048
    hop_length=512
    num_segments=5

    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_number_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for index, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        if dirpath is not dataset_path:
            
            dirpath_components = dirpath.split("\\")
            genre_label = dirpath_components[-1]
            data["mapping"].append(genre_label)
            print("\nProcessing {}".format(genre_label))

            for f in filenames:

                file_path = os.path.join(dirpath, f)
                signal, samp_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    try:
                        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=samp_rate, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                    
                        mfcc = mfcc.T

                        if len(mfcc) == expected_number_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(index - 1)
                            print("{}, segment:{}".format(file_path, s + 1))
                    except Exception as e:
                        continue

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ =="__main__":
    main()