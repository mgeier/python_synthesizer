#!pypy3
import threading

import numpy as np
import sounddevice
import mido


amplitude = 0.1
attack_seconds = 0.01
release_seconds = 0.2


def m2f(note):
    """Convert MIDI note number to frequency in Hertz.

    See https://en.wikipedia.org/wiki/MIDI_Tuning_Standard.

    """
    return 2 ** ((note - 69) / 12) * 440


def update_envelope(envelope, begin, target, vel):
    """Helper function to calculate envelopes.

    envelope: array of velocities, will be mutated
    begin: sample index where ramp begins
    target: sample index where *vel* shall be reached
    vel: final velocity value
    If the ramp goes beyond the blocksize, it is supposed to be
    continued in the next block.
    A reference to *envelope* is returned, as well as the (unchanged)
    *vel* and the target index of the following block where *vel* shall
    be reached.

    """
    blocksize = len(envelope)
    old_vel = envelope[begin]
    slope = (vel - old_vel) / (target - begin + 1)
    ramp = np.arange(min(target, blocksize) - begin) + 1
    envelope[begin:target] = ramp * slope + old_vel
    if target < blocksize:
        envelope[target:] = vel
        target = 0
    else:
        target -= blocksize
    return envelope, vel, target


def play_midi_file(path, samplerate=48000, blocksize=1024, **kwargs):
    assert blocksize > 0

    midi_stream = iter(mido.MidiFile(path))

    for msg in midi_stream:
        if msg.type == 'note_on':
            break
        print('ignoring', msg.dict())

    offset = 0
    current_frame = 0
    voices = {}

    attack = int(attack_seconds * samplerate)
    release = int(release_seconds * samplerate)

    def callback(outdata, frames, time, status):
        nonlocal msg
        nonlocal offset
        nonlocal current_frame

        assert blocksize == frames
        if status:
            print(status)

        # Step 1: Update/delete existing voices from previous block

        # Iterating over a copy because items may be deleted:
        for pitch in list(voices):
            envelope, vel, target = voices[pitch]
            if any([vel, target]):
                envelope[0] = envelope[-1]
                voices[pitch] = update_envelope(envelope, 0, target, vel)
            else:
                del voices[pitch]

        # Step 2: Create envelopes from the MIDI events of the current block

        if offset is None:
            if not voices:
                outdata.fill(0)
                raise sounddevice.CallbackAbort()
        else:
            while offset < frames:
                if msg.type == 'note_on' and msg.velocity > 0:
                    try:
                        envelope, _, _ = voices[msg.note]
                    except KeyError:
                        envelope = np.zeros(frames)
                    voices[msg.note] = update_envelope(
                        envelope, offset, offset + attack, msg.velocity)
                elif msg.type in ('note_on', 'note_off'):
                    # NoteOff velocity is ignored!
                    try:
                        envelope, _, _ = voices[msg.note]
                    except KeyError:
                        print('NoteOff without NoteOn (ignored)')
                    else:
                        voices[msg.note] = update_envelope(
                            envelope, offset, offset + release, 0)
                else:
                    print('ignoring', msg.dict())

                try:
                    msg = next(midi_stream)
                except StopIteration:
                    offset = None
                    break
                offset += round(msg.time * samplerate)
            if offset is not None:
                offset -= frames

        # Step 3: Create sine tones, apply envelopes, add to output buffer

        outdata.fill(0)
        for pitch, (envelope, _, _) in voices.items():
            t = (np.arange(frames) + current_frame) / samplerate
            tone = amplitude * sum(
                1 / (i ** 1.5) * np.sin(2 * np.pi * m2f(pitch) * i * t)
                for i in range(1, 8)
            )
            outdata[:, 0] += tone * envelope / 127
        current_frame += frames

    event = threading.Event()

    stream = sounddevice.OutputStream(
        channels=1,
        blocksize=blocksize,
        callback=callback,
        samplerate=samplerate,
        finished_callback=event.set,
        **kwargs)
    with stream:
        event.wait()


if __name__ == '__main__':
    play_midi_file('FurElise.mid')
