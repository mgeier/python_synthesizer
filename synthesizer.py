import mido
import numpy as np
import rtmixer
import sounddevice as sd

MASTER_VOLUME = 0.1


class Player(rtmixer.Mixer):

    def play_note(self, frequency, *, time, volume=1):
        overtones = {i: 1 / (i ** 1.5) for i in range(1, 8)}
        half_life = 0.3
        duration = 1.5
        fs = self.samplerate
        t = np.arange(int(duration * fs), dtype='float32') / fs
        buffer = sum(
            overtone_volume * np.sin(2 * np.pi * t * frequency * overtone)
            for overtone, overtone_volume in overtones.items())
        buffer *= MASTER_VOLUME * volume * 2 ** (-t / half_life)
        while len(self.actions) > 20:
            # Don't calculate all notes at once (apply backpressure)
            sd.sleep(100)
        assert buffer.dtype == np.float32
        assert buffer.flags.c_contiguous == True
        return self.play_buffer(
            buffer, channels=1, start=time, allow_belated=False)


def play_midi(path):
    midifile = mido.MidiFile(path)
    with Player(channels=1) as p:
        delay = 0.5
        current_time = p.time + delay
        for message in midifile:
            current_time += message.time
            if message.type != 'note_on':
                continue
            p.play_note(
                time=current_time,
                frequency=220 * 2 ** ((message.note - 57) / 12),
                volume=message.velocity / 127
            )
        p.wait()
    print(p.stats.output_underflows, 'output underflows')


if __name__ == '__main__':
    play_midi('FurElise.mid')
