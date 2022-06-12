import librosa
import numpy as np
import sounddevice as sd

from const import *


sd.default.samplerate = SR
sd.default.channels = CHANNELS
sd.default.blocksize = BLOCKSIZE

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

    def get_track(self):
        return self.track[:self.len]

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
            return max(t.len for t in tracks)

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

    def get_track(self, number):
        try:
            return self.tracks[number].get_track()
        except KeyError:
            raise PlayerException("Empty track.")

    def export(self, min_length):
        length = (1 + min_length * SR // self.length) * self.length

        frames = BLOCKSIZE
        output = None
        progress = 0
        while progress < length:
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
