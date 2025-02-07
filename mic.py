import alsaaudio
import threading
import numpy as np

class AudioCapture(threading.Thread):
    def __init__(self, output_audio):
        threading.Thread.__init__(self)
        self.output_audio = output_audio
        self.running = False
        
    def run(self, periodsize=160):
        # Settings compatible with Whisper
        inp = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            channels=1,
            rate=16000,
            format=alsaaudio.PCM_FORMAT_FLOAT_LE,
            periodsize=periodsize
        )
        
        self.running = True
        while self.running:
            length, data = inp.read()
            if length > 0:
                self.output_audio(np.frombuffer(data, dtype=np.float32))
        
    def stop(self):
        self.running = False

