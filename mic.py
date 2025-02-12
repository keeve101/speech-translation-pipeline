import alsaaudio
import threading
import numpy as np

def process_audio(buffer, gain=16.0):
    buf = np.frombuffer(buffer, dtype=np.int32)
    buf = buf.astype(np.float32)/float(2**31)
    buf *= gain
    return np.clip(buf, -1, 1)

class AudioCapture(threading.Thread):
    def __init__(self, output_audio):
        threading.Thread.__init__(self)
        self.output_audio = output_audio
        self.running = False
        
    def run(self, periodsize=1600):
        # Settings compatible with Whisper
        inp = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            channels=1,
            rate=16000,
            format=alsaaudio.PCM_FORMAT_S32_LE,
            periodsize=periodsize
        )
        
        self.running = True
        while self.running:
            length, data = inp.read()
            if length > 0:
                self.output_audio(process_audio(data))
        
    def stop(self):
        self.running = False

