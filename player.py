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
        self.volume = 1
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

    def get_reference_progress(self, exclude=None):
        tracks = [t for k, t in self.tracks.items() if k != exclude and t.playing]
        if len(tracks) == 0:
            return 0
        else:
            return max(t.progress - t.frames for t in tracks)

    def get_max_progress(self, exclude=None):
        tracks = [t for k, t in self.tracks.items() if k != exclude and t.playing]
        progress = 0
        _len = 0
        for t in tracks:
            if t.len > _len:
                _len = t.len
                progress = t.progress

        return progress

    def get_max_frame(self, exclude=None):
        tracks = [t for k, t in self.tracks.items() if k != exclude]
        if len(tracks) == 0:
            return None
        else:
            return max(t.len for t in tracks)

    def get_reference_frame(self, exclude=None):
        tracks = [t for k, t in self.tracks.items() if k != exclude]
        if len(tracks) == 0:
            return None
        else:
            return min(t.len for t in tracks)

    @property
    def length(self):
        if len(self.tracks) == 0:
            return None
        else:
            return max(t.len for t in self.tracks.values())

    def add_track(self, n, track):
        self.tracks[n] = Track(track)

    def remove_track(self, n):
        try:
            del self.tracks[n]
        except KeyError:
            raise PlayerException("Cannot remove empty track.")

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

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

    def info(self):
        if not self.tracks:
            return
        length = max(t.len for t in self.tracks.values()) // SR
        print(length)
        output = self.export(length)
        print (max(output))
        print (min(output))
        print (np.average(output))

    def export(self, length=30):
        frames = 5000
        output = None
        progress = 0
        while progress < length * SR:
            chunks = []
            for track in self.tracks.values():
                a = progress % track.len
                b = a + frames
                chunk = track.track[a:b]
                chunk = track.volume * chunk
                chunks.append(chunk)
                _sum = sum(chunks)
            if output is None:
                output = _sum
            else:
                output = np.concatenate((output, _sum))
            progress += frames
        return output

    def callback(self, outdata, frames, time, status):
        chunks = []
        for track in self.tracks.values():
            if track.playing:
                a = self.play_index % track.len
                b = a + frames
                chunk = track.track[a:b]
                chunk = track.volume * chunk
                chunks.append(chunk)
                track.progress = a
                track.frames = frames
        if chunks:
            outdata[:] = sum(chunks)
        else:
            outdata[:] = np.zeros(shape=(frames, 1))
        self.play_index += frames
