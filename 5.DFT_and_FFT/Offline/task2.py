import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from discrete_framework import DFTAnalyzer, DiscreteSignal, FastFourierAnalyzer

class AudioEqualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("DFT Audio Equalizer")
        
        self.samplerate = 0
        self.original_audio = None
        self.processed_audio = None
        
        # --- UI Layout ---
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)
        
        tk.Button(top_frame, text="Load WAV File", command=self.load_file).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Process & Play", command=self.process_and_play).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Stop Audio", command=sd.stop).pack(side=tk.LEFT, padx=10)
        
        # Toggle Switch
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text="Algorithm: ").pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="DFT (Slow)", variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT (Fast)", variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # Equalizer Sliders
        self.slider_frame = tk.Frame(root)
        self.slider_frame.pack(pady=20, padx=20)
        
        self.sliders = []
        labels = ["Low", "Low-Mid", "Mid", "High-Mid", "High"]
        for i in range(5):
            frame = tk.Frame(self.slider_frame)
            frame.pack(side=tk.LEFT, padx=5)
            tk.Label(frame, text=labels[i], font=("Arial", 8)).pack()
            slider = tk.Scale(frame, from_=2.0, to=0.0, resolution=0.1, length=150, orient=tk.VERTICAL)
            slider.set(1.0)
            slider.pack()
            self.sliders.append(slider)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            try:
                self.samplerate, data = wav.read(file_path)
                
                # Normalize to float [-1, 1]
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0
                
                # If already float, just ensure float32
                if data.dtype != np.float32:
                    data = data.astype(np.float32)

                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                self.original_audio = data
                self.processed_audio = None
                duration = len(data) / self.samplerate
                print(f"Loaded: {len(data)} samples, {self.samplerate} Hz, {duration:.1f}s")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")

    def process_and_play(self):
        if self.original_audio is None:
            messagebox.showwarning("Warning", "Please load a WAV file first.")
            return
        
        print("Starting processing...")
        # Get Slider Values
        gains = [s.get() for s in self.sliders]
        
        # TODO: Implement the chunking, FFT, filtering, IFFT, and overlap-add here.
        
        # For starter code, we just play the original audio so the button "works"
        # In the final version, this should play self.processed_audio

        audio=self.original_audio
        if self.use_fft.get():
            
            analyzer = FastFourierAnalyzer()
        else:
            analyzer = DFTAnalyzer()

        chunk_size = 1024

        audio_signal = DiscreteSignal(audio)
        audio_padded = audio_signal.pad((len(audio) + chunk_size - 1) // chunk_size * chunk_size).data

        num_chunks = len(audio_padded) // chunk_size
        output_audio = np.zeros_like(audio_padded)


        half_N=chunk_size//2
        band_edges=[0,half_N//5,half_N*2//5,half_N*3//5,half_N*4//5,half_N]

        for i in range(num_chunks):
            start_idx=i*chunk_size
            end_idx= (i+1)*chunk_size

            chunk= DiscreteSignal(audio_padded[start_idx:end_idx])
            spectrum=analyzer.compute_dft(chunk)

            for j in range(5):
                gain_factor=gains[j]

                start_bin=band_edges[j]
                end_bin=band_edges[j+1]

                spectrum[start_bin:end_bin]*=gain_factor

                if start_bin ==0:
                    spectrum[chunk_size - end_bin + 1 : chunk_size] *= gain_factor

                else:
                    start_neg = chunk_size - end_bin
                    end_neg = chunk_size - start_bin
                    spectrum[start_neg:end_neg] *= gain_factor

            filtered_chunk=analyzer.compute_idft(spectrum)
            output_audio[start_idx:end_idx]=np.real(filtered_chunk)

        output_audio=output_audio[:len(audio)]
 
        self.processed_audio = output_audio 
    
        sd.stop()
        #default_output = sd.default.device[0]
        sd.play(np.real(output_audio), self.samplerate)

        print("Playing processed audio...")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEqualizer(root)
    root.mainloop()