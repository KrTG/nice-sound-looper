import librosa
import numpy as np
import sounddevice as sd

SR = 48000
FFT_WIDTH = 2048
HOP_LENGTH = 512
CHANNELS = 1

sd.default.samplerate = SR
sd.default.channels = CHANNELS


class PlayerException(Exception):
    pass

class Track():
    # Track is doubled for easy looping
    def __init__(self, track):
        self.len = track.shape[0]
        self.track = np.tile(track, (2, 1))
        self.playing = False
        self.progress = 0
        self.frames = 0

    def cut(self, amount):
        track = self.track[:self.len]
        track = track[:-amount]
        spect = librosa.feature.melspectrogram(y=track.flatten(), sr=SR, n_mels=128, fmax=10000, hop_length=HOP_LENGTH)
        self.len = track.shape[0]
        self.track = np.tile(track, (2, 1))

        return spect

class Player():
    def __init__(self):
        self.tracks = {}
        self.stream = sd.OutputStream(callback=self.callback)
        self.play_index = 0
        self._next_chunk_size = 0

    @property
    def reference_progress(self):
        if len(self.tracks) == 0:
            return 0
        else:
            print(max(t.progress - t.frames for t in self.tracks.values()) / SR)
            print(max(t.progress - t.len - t.frames for t in self.tracks.values()) / SR)
            return max(t.progress - t.frames for t in self.tracks.values())

    @property
    def reference_frame(self):
        if len(self.tracks) == 0:
            return None
        else:
            return min(t.len for t in self.tracks.values())

    @property
    def length(self):
        if len(self.tracks) == 0:
            return None
        else:
            return max(t.len for t in self.tracks.values())

    def add_track(self, n, track):
        self.tracks[n] = Track(track)

    def start(self):
        self.stream.start()

    def play(self, n):
        try:
            self.tracks[n].playing = True
        except KeyError:
            raise PlayerException("Empty track cannot play.")

    def pause(self, n):
        try:
            self.tracks[n].playing = False
        except KeyError:
            raise PlayerException("Empty track.")

    def cut(self, n):
        if self.tracks[n].len <= self.reference_frame:
            raise PlayerException("Cannot cut longest track.")
        try:
            return self.tracks[n].cut(self.reference_frame)
        except KeyError:
            raise PlayerException("Cannot cut empty track.")

    def callback(self, outdata, frames, time, status):
        chunks = []
        for track in self.tracks.values():
            if track.playing:
                a = self.play_index % track.len
                b = a + frames
                chunks.append(track.track[a:b])
                track.progress = a
                track.frames = frames
        if chunks:
            outdata[:] = sum(chunks)
        else:
            outdata[:] = np.zeros(shape=(frames, 1))
        self.play_index += frames
