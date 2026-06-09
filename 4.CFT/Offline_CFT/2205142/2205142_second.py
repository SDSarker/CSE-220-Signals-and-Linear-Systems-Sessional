from os import path
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# =====================================================
# Continuous Image Class
# =====================================================
class ContinuousImage:
    """
    Represents an image as a continuous 2D signal.
    """

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L')
        self.image = self.image / np.max(self.image)

        # Define continuous spatial axes
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])

    def show(self, title="Image"):
        plt.imshow(self.image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()


# =====================================================
# 2D Continuous Fourier Transform Class
# =====================================================
class CFT2D:
    """
    Computes 2D Continuous Fourier Transform
    using separability and numerical integration.
    """

    def __init__(self, image_obj:ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

    def compute_cft(self):
        """
        Compute real and imaginary parts of 2D CFT.
        """

        real = np.zeros_like(self.I)
        img = np.zeros_like(self.I)

        

        for i in range(len(self.x)):
            for j in range(len(self.y)):
                
                ui=self.x[i]
                vj=self.y[j]
                print(f"u: {i}/{len(self.x)} v: {j}/{len(self.y)}")
                cos_x= np.cos(2*np.pi*(ui*self.x[:, None]+vj*self.y[None,:]))
                sin_x= np.sin(2*np.pi*(ui*self.x[:, None]+vj*self.y[None,:]))

                real[i,j]=np.trapezoid(np.trapezoid(self.I*cos_x,self.y),self.x)
                img[i,j]= -np.trapezoid(np.trapezoid(self.I*sin_x,self.y),self.x)


        return real,img


    def plot_magnitude(self):
        """
        Plot log-scaled magnitude spectrum.
        """
        real, img = self.compute_cft()
        magnitude = np.sqrt(real**2 + img**2)
        plt.imshow(np.log(1 + magnitude), cmap='magma',extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        plt.title("2D Magnitude Spectrum")
        
        plt.axis('off')
        plt.show()
        


# =====================================================
# Frequency Filtering
# =====================================================
class FrequencyFilter:
    def low_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows//2, cols//2

        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i-cx)**2 + (j-cy)**2) > cutoff:
                    real[i,j] = 0
                    imag[i,j] = 0
        return real, imag

# =====================================================
# Inverse 2D Continuous Fourier Transform
# =====================================================
class InverseCFT2D:
    """
    Reconstructs image from 2D frequency spectrum.
    """

    def __init__(self, real, imag, x, y):
        self.real=real 
        self.imag = imag
        self.x = x
        self.y = y

    def reconstruct(self):
        """
        Perform inverse 2D CFT using numerical integration.
        """
        rows=self.real.shape[1]
        cols=self.real.shape[0]
        reconstructed = np.zeros_like(self.real)

        U, V = np.meshgrid(self.x, self.y, indexing="xy")

        for j in range(rows):
            yj=self.y[j]
            for i in range(cols):
                xi=self.x[i]

                arg=2 * np.pi * (U * xi + V * yj)
                integrand = self.real * np.cos(arg) - self.imag * np.sin(arg)

                print(f"u: {i}/{len(self.x)} v: {j}/{len(self.y)}")
                temp=np.trapezoid(np.trapezoid(integrand, self.y, axis=0), self.x, axis=0)
                reconstructed[i, j] = temp
        
        
        return reconstructed


# =====================================================
# Main Execution (Task 2)
# =====================================================
img = ContinuousImage("noisy_image.png")
img.show("Original Image")

cft2d = CFT2D(img)
real, imag = cft2d.compute_cft()
cft2d.plot_magnitude()

filt = FrequencyFilter()
real_f, imag_f = filt.low_pass(real, imag, cutoff=40)

icft2d = InverseCFT2D(real_f, imag_f, img.x, img.y)
denoised = icft2d.reconstruct()

plt.imshow(denoised, cmap='gray')
plt.title("Reconstructed (Denoised) Image")
plt.axis('off')
plt.show()
