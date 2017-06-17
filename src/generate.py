#!/usr/bin/env python
"""
Generates the audio file with the trained RNN
"""
import time
import logging
import numpy as np
import librosa
from tqdm import tqdm
from scipy.spatial.distance import cdist

from keras.models import model_from_json


NORM_CQT_FRAMES = "../data/norm_cqt_frames.npy"
AUDIO_FRAMES_NPY = "../data/audio_frames.npy"
N_FRAMES = 100  # Must be the same as in train.py
N_GEN_FRAMES = 1000  # Number of frames to generate
SR = 22050
HOP_LEN = 4096
FRAME_LEN = HOP_LEN * 2  # 50% overlap


def get_closest_frames(in_frames, all_cqt_frames):
    out_frames = []
    for in_frame in tqdm(in_frames):
        dists = cdist(in_frame.reshape((1, -1)), all_cqt_frames).flatten()
        out_frames.append(np.argsort(dists)[0])
    return np.asarray(out_frames, dtype=np.int)


def process():
    logging.info("Loading model from disk")
    with open('model.json', 'r') as f:
        model_json = f.read()
        model = model_from_json(model_json)
    model.load_weights("model.h5")

    # Load normalized frames
    frames = np.load(NORM_CQT_FRAMES)

    # Generate frames
    logging.info("Predicting Frames")
    # TODO: Randomize
    start_i = 50
    seed_frames = frames[start_i:start_i + N_FRAMES]
    pred_frames = [frame for frame in seed_frames]
    for i in tqdm(range(N_GEN_FRAMES)):
        preds = model.predict(seed_frames.reshape((1, N_FRAMES, frames.shape[1])),
                              verbose=0)

        # Update the seed_frames
        seed_frames = seed_frames[1:]
        seed_frames = np.concatenate((seed_frames, preds), axis=0)

        # Store the predicted frame
        pred_frames.append(preds.flatten())
    pred_frames = np.asarray(pred_frames)

    # Get closest frames to map to audio
    audio_idxs = get_closest_frames(pred_frames, frames)
    print(audio_idxs)

    # Generate audio
    logging.info("Generating audio")
    all_audio_frames = np.load(AUDIO_FRAMES_NPY)
    audio = np.zeros(len(audio_idxs) * HOP_LEN + FRAME_LEN)
    for i, audio_idx in enumerate(audio_idxs):
        audio_frame = np.hanning(FRAME_LEN) * all_audio_frames[audio_idx]
        audio[i * HOP_LEN: i * HOP_LEN + FRAME_LEN] += audio_frame
    librosa.output.write_wav("out.wav", audio, sr=SR)


if __name__ == "__main__":
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)
    process()
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))
