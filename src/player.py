import numpy as np
import sounddevice as sd

from src.const import *

sd.default.samplerate = SR
sd.default.channels = CHANNELS
sd.default.blocksize = BLOCKSIZE


class PlayerException(Exception):
    pass


class Track:
    # Track is doubled for easy looping
    def __init__(self, track):
        self.len = track.shape[0]
        self.track = np.tile(track, (2, 1))
        self.playing = False
        self.volume = 1
        self.progress = 0
        self.frames = 0

    def get_track(self):
        return self.track[: self.len]


class Player:
    def __init__(self):
        self.tracks = {}
        self.stream = sd.OutputStream(callback=self.callback)
        self.time_index = 0

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

    def change_startpoint(self, percent_change):
        frame = self.get_max_frame()
        change = int(frame * percent_change)
        for track in self.tracks.values():
            adjustment = change % track.len
            track.track = np.concatenate(
                (track.track[adjustment:], track.track[:adjustment])
            )

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
        except KeyError as e:
            raise PlayerException("Cannot remove empty track.") from e

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

    def play(self, n):
        try:
            self.tracks[n].playing = True
        except KeyError as e:
            raise PlayerException("Empty track cannot play.") from e

    def pause(self, n):
        try:
            self.tracks[n].playing = False
        except KeyError as e:
            raise PlayerException("Empty track.") from e

    def forward_to(self, x):
        self.time_index = x

    def get_track(self, number):
        try:
            return self.tracks[number].get_track()
        except KeyError as e:
            raise PlayerException("Empty track.") from e

    def export(self, min_length):
        length = (1 + min_length * SR // self.length) * self.length

        frames = BLOCKSIZE
        output = None
        progress = 0
        while progress < length:
            chunks = []
            for track in self.tracks.values():
                if track.playing:
                    a = progress % track.len
                    b = a + frames
                    chunk = track.track[a:b]
                    chunk = track.volume * chunk
                    chunks.append(chunk)
            if chunks:
                _sum = sum(chunks)
            else:
                _sum = np.zeros(shape=(frames, CHANNELS))
            if output is None:
                output = _sum
            else:
                output = np.concatenate((output, _sum))
            progress += frames
        return output

    def callback(self, outdata, frames, _time, _status):
        chunks = []
        for track in self.tracks.values():
            if track.playing:
                a = self.time_index % track.len
                b = a + frames
                chunk = track.track[a:b]
                chunk = track.volume * chunk
                chunks.append(chunk)
                track.progress = a
                track.frames = frames
        if chunks:
            outdata[:] = sum(chunks)
        else:
            outdata[:] = np.zeros(shape=(frames, CHANNELS))
        self.time_index += frames
