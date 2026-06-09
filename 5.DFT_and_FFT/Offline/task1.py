import tkinter as tk
import numpy as np
import math
from discrete_framework import DiscreteSignal, DFTAnalyzer, FastFourierAnalyzer

class DoodlingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Epicycles Doodler")
        
        # --- UI Layout ---
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()
        
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)
        
        # Buttons
        tk.Button(control_frame, text="Clear Canvas", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Draw Epicycles", command=self.run_transform).pack(side=tk.LEFT, padx=5)
        
        # Toggle Switch (Radio Buttons)
        self.use_fft = tk.BooleanVar(value=False)
        tk.Label(control_frame, text=" |  Algorithm: ").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(control_frame, text="Naive DFT", variable=self.use_fft, value=False).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="FFT", variable=self.use_fft, value=True).pack(side=tk.LEFT)

        # State Variables
        self.points = []
        self.drawing = False
        self.fourier_coeffs = None
        self.is_animating = False
        self.after_id = None

        # Bindings
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def start_draw(self, event):
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.canvas.delete("all")
        self.points = []
        self.drawing = True

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.points.append((x, y))
            r = 2
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")

    def end_draw(self, event):
        self.drawing = False

    def clear(self):
        self.canvas.delete("all")
        self.points = []
        self.is_animating = False
        if self.after_id:
            self.root.after_cancel(self.after_id)

    def draw_epicycle(self, x, y, radius):
        """
        Helper method for students to draw a circle (epicycle).
        x, y: Center coordinates
        radius: Radius of the circle
        """
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline="blue", tags="epicycle")

    def run_transform(self):
        if not self.points: return
        
        # TODO: Implementation
        # 1. Convert (x,y) points to Complex Signal
        # 2. Select Algorithm 
        # 3. Compute Transform

        #1

        points = [x + 1j * y for x, y in self.points]
        complex_signal=DiscreteSignal(points)

        #2

        if self.use_fft.get():
            analyzer=FastFourierAnalyzer()
            N_orig = len(complex_signal)
            next_p2 = 1 << (N_orig - 1).bit_length()
            complex_signal = complex_signal.interpolate(next_p2)
            
            
        else:
            analyzer=DFTAnalyzer()

        #3

        self.X=analyzer.compute_dft(complex_signal)
        self.N = len(complex_signal)

        
        self.k_sorted=np.argsort(np.abs(self.X))[::-1]
        
        # print("Computed DFT Coefficients (sorted by magnitude):")
        # for k in self.k_sorted:
        #     print(f"  k={k}: {self.X[k]}")
        self.animate_epicycles(center_offset=0)

    def animate_epicycles(self, center_offset):
        self.is_animating = True
        self.time_step = 0
        self.num_frames = self.N
        self.path=[]

        
        self.center_offset = center_offset
        self.update_frame()

    def update_frame(self):
        if not self.is_animating: return
        
        self.canvas.delete("epicycle") 
        
        # TODO: Implementation
        # 1. Calculate the current time 't' based on self.time_step
        # 2. Reconstruct the signal value 'z' at time 't' 
        # 3. Draw the epicycles:
        # 4. Draw the tips

        #1
        t=self.time_step

        #2
        x=0
        y=0


        for k in self.k_sorted:

            prev_x,prev_y=x,y

            X_k=self.X[k]/self.N

            phase = 2 * math.pi * k * t / self.N
            v= X_k * np.exp(1j*phase)

            x+=v.real
            y+=v.imag

            radius=np.abs(X_k)
            
            #3

            if k==0  : continue

            if radius>0.5:
                self.draw_epicycle(prev_x, prev_y, radius)

            self.canvas.create_line(prev_x, prev_y, x, y, fill="gray", tags="epicycle")
        

        #4

        self.path.append((x,y))
        for i in range(1, len(self.path)):
            x1, y1 = self.path[i-1]
            x2, y2 = self.path[i]
            self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2, tags="epicycle")

        self.time_step = (self.time_step + 1)

        # Loop animation

        if self.time_step >= self.num_frames:
            self.time_step = 0
            self.path = []
        
        self.after_id = self.root.after(50, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = DoodlingApp(root)
    root.mainloop()