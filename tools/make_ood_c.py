# -*- coding: utf-8 -*-
"""Create corrupted copies of an image folder with the 15 ImageNet-C corruptions.

Adapted from https://github.com/hendrycks/robustness (Apache-2.0).
Output layout: <output_dir>/<corruption>/<severity>/<image>.
"""

import os
from PIL import Image
import os.path
import time
import numpy as np
import glob
import collections

# NumPy 2.0 compatibility: reintroduce removed aliases if missing
for _name, _rep in [('float_', np.float64)]:
    if not hasattr(np, _name):
        setattr(np, _name, _rep)

# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import sys
import argparse
from tqdm import tqdm
import concurrent.futures
# Import wand libraries conditionally (some environments might not have it)
try:
    from wand.image import Image as WandImage
    from wand.api import library as wandlibrary
    import wand.color as WandColor
    import ctypes
    
    # Tell Python about the C method
    wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                                  ctypes.c_double,  # radius
                                                  ctypes.c_double,  # sigma
                                                  ctypes.c_double)  # angle

    # Extend wand.image.Image class to include method signature
    class MotionImage(WandImage):
        def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
            wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
            
    WAND_AVAILABLE = True
except ImportError:
    print("Warning: Wand library not available. Motion blur, snow, and some other corruptions will not work.")
    WAND_AVAILABLE = False

from PIL import Image as PILImage

warnings.simplefilter("ignore", UserWarning)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)  # replaced np.float_
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size (32x32)
        c = [0.04, 0.06, .08, .09, .10][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size (64x64)
        c = [0.04, 0.08, .12, .15, .18][severity - 1]
    else:  # ImageNet size (224x224 or larger)
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [500, 250, 100, 75, 50][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [250, 100, 50, 30, 15][severity - 1]
    else:  # ImageNet size
        c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [.01, .02, .03, .05, .07][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [.01, .02, .05, .08, .14][severity - 1]
    else:  # ImageNet size
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [.06, .1, .12, .16, .2][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [.15, .2, 0.25, 0.3, 0.35][severity - 1]
    else:  # ImageNet size
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [.4, .6, 0.7, .8, 1][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [.5, .75, 1, 1.25, 1.5][severity - 1]
    else:  # ImageNet size
        c = [1, 2, 3, 4, 6][severity - 1]
        
    # Handle channel_axis parameter name change in different skimage versions
    try:
        x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    except TypeError:
        # For older versions of skimage
        x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
        
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]
        h, w = x.shape[:2]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [(0.1,1,1), (0.5,1,1), (0.6,1,2), (0.7,2,1), (0.9,2,2)][severity - 1]
        h, w = 64, 64
    else:  # ImageNet size
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        h, w = 224, 224
    
    # Handle channel_axis parameter name change in different skimage versions
    try:
        x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)
    except TypeError:
        # For older versions of skimage
        x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h_idx in range(h - c[1], c[1], -1):
            for w_idx in range(w - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h_idx + dy, w_idx + dx
                # swap
                x[h_idx, w_idx], x[h_prime, w_prime] = x[h_prime, w_prime], x[h_idx, w_idx]

    # Handle channel_axis parameter name change in different skimage versions
    try:
        return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255
    except TypeError:
        # For older versions of skimage
        return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [(0.5, 0.6), (1, 0.1), (1.5, 0.1), (2.5, 0.01), (3, 0.1)][severity - 1]
    else:  # ImageNet size
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3xHxW -> HxWx3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1, image_size=32):
    if not WAND_AVAILABLE:
        print("Warning: Wand library not available. Motion blur not applied.")
        return np.array(x)
        
    if image_size < 64:  # CIFAR-10 size
        c = [(6,1), (6,1.5), (6,2), (8,2), (9,2.5)][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [(10,1), (10,1.5), (10,2), (10,2.5), (12,3)][severity - 1]
    else:  # ImageNet size
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)
    
    # Normalize the decoded image to 3-channel RGB
    if len(x.shape) == 3 and x.shape[2] == 3:  # color image
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    elif len(x.shape) == 3 and x.shape[2] == 4:  # image with alpha channel
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB, drop alpha
    elif len(x.shape) == 2:  # grayscale image
        # grayscale -> 3-channel RGB
        x_rgb = np.stack([x, x, x], axis=2)
        return np.clip(x_rgb, 0, 255)
    else:
        # unexpected layout
        print(f"Warning: Unexpected image shape {x.shape} in motion_blur. Converting to RGB.")
        if len(x.shape) > 2:
            # use the first channel as grayscale
            gray = x[:, :, 0] if x.shape[2] > 0 else x[:, :, 0]
            x_rgb = np.stack([gray, gray, gray], axis=2)
            return np.clip(x_rgb, 0, 255)
        else:
            return np.array(x)

def zoom_blur(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]
    else:  # ImageNet size
        c = [np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01), np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02), np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][severity - 1]
        mapsize = 32
    elif image_size < 224:  # Tiny-ImageNet size
        c = [(.4,3), (.7,3), (1,2.5), (1.5,2), (2,1.75)][severity - 1]
        mapsize = 64
    else:  # ImageNet size
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
        mapsize = 256

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(mapsize=mapsize, wibbledecay=c[1])[:image_size, :image_size][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
        resize_factor = 0.2
    elif image_size < 224:  # Tiny-ImageNet size
        c = [(1, 0.3), (0.9, 0.4), (0.8, 0.45), (0.75, 0.5), (0.7, 0.55)][severity - 1]
        resize_factor = 0.3
    else:  # ImageNet size
        c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
        resize_factor = 1.0

    # Frost textures from the ImageNet-C repository (tools/frost/)
    frost_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frost')
    frost_files = [os.path.join(frost_dir, f) for f in
                   ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpg', 'frost5.jpg', 'frost6.jpg']]
    
    # Check if frost files exist
    frost_file_exists = False
    for file in frost_files:
        if os.path.exists(file):
            frost_file_exists = True
            break
            
    if not frost_file_exists:
        print("Warning: Frost files not found. Frost effect will not be applied.")
        return np.array(x)
    
    idx = np.random.randint(5)
    filename = frost_files[idx]
    frost = cv2.imread(filename)
    
    if frost is None:
        print(f"Warning: Could not read frost file {filename}. Frost effect will not be applied.")
        return np.array(x)
        
    frost = cv2.resize(frost, (0, 0), fx=resize_factor, fy=resize_factor)
    
    # Check if the frost image is large enough
    if frost.shape[0] < image_size or frost.shape[1] < image_size:
        print(f"Warning: Frost image too small ({frost.shape}) for the input size ({image_size}x{image_size}). Frost effect will not be applied.")
        return np.array(x)
    
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - image_size), np.random.randint(0, frost.shape[1] - image_size)
    frost = frost[x_start:x_start + image_size, y_start:y_start + image_size][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1, image_size=32):
    if not WAND_AVAILABLE:
        print("Warning: Wand library not available. Snow effect will not be applied.")
        return np.array(x)
        
    if image_size < 64:  # CIFAR-10 size
        c = [(0.1,0.2,1,0.6,8,3,0.95),
             (0.1,0.2,1,0.5,10,4,0.9),
             (0.15,0.3,1.75,0.55,10,4,0.9),
             (0.25,0.3,2.25,0.6,12,6,0.85),
             (0.3,0.3,1.25,0.65,14,12,0.8)][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [(0.1,0.2,1,0.6,8,3,0.8),
             (0.1,0.2,1,0.5,10,4,0.8),
             (0.15,0.3,1.75,0.55,10,4,0.7),
             (0.25,0.3,2.25,0.6,12,6,0.65),
             (0.3,0.3,1.25,0.65,14,12,0.6)][severity - 1]
    else:  # ImageNet size
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(image_size, image_size, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [(0.62,0.1,0.7,0.7,0.5,0),
             (0.65,0.1,0.8,0.7,0.5,0),
             (0.65,0.3,1,0.69,0.5,0),
             (0.65,0.1,0.7,0.69,0.6,1),
             (0.65,0.1,0.5,0.68,0.6,1)][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [(0.62,0.1,0.7,0.7,0.6,0),
             (0.65,0.1,0.8,0.7,0.6,0),
             (0.65,0.3,1,0.69,0.6,0),
             (0.65,0.1,0.7,0.68,0.6,1),
             (0.65,0.1,0.5,0.67,0.6,1)][severity - 1]
    else:  # ImageNet size
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
             (0.65, 0.3, 3, 0.68, 0.6, 0),
             (0.65, 0.3, 2, 0.68, 0.5, 0),
             (0.65, 0.3, 1, 0.65, 1.5, 1),
             (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
             
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    # Handle channel_axis parameter name change in different skimage versions
    try:
        liquid_layer = gaussian(liquid_layer, sigma=c[2])
    except TypeError:
        # For older versions of skimage
        liquid_layer = gaussian(liquid_layer, sigma=c[2], multichannel=False)
        
    liquid_layer[liquid_layer < c[3]] = 0
    
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        
        # Handle channel_axis parameter name change in different skimage versions
        try:
            m = gaussian(m.astype(np.float32), sigma=c[4])
        except TypeError:
            # For older versions of skimage
            m = gaussian(m.astype(np.float32), sigma=c[4], multichannel=False)
            
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [.75, .5, .4, .3, 0.15][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [.4, .3, .2, .1, 0.05][severity - 1]
    else:  # ImageNet size
        c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [.05, .1, .15, .2, .3][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [.1, .2, .3, .4, .5][severity - 1]
    else:  # ImageNet size
        c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (30, 0.2)][severity - 1]
    else:  # ImageNet size
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [80, 65, 58, 50, 40][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [65, 58, 50, 40, 25][severity - 1]
    else:  # ImageNet size
        c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [0.9, 0.8, 0.7, 0.6, 0.5][severity - 1]
    else:  # ImageNet size
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = x.resize((int(image_size * c), int(image_size * c)), PILImage.BOX)
    x = x.resize((image_size, image_size), PILImage.BOX)

    return x


def elastic_transform(image, severity=1, image_size=32):
    if image_size < 64:  # CIFAR-10 size
        c = [(image_size*0, image_size*0, image_size*0.08),
             (image_size*0.05, image_size*0.2, image_size*0.07),
             (image_size*0.08, image_size*0.06, image_size*0.06),
             (image_size*0.1, image_size*0.04, image_size*0.05),
             (image_size*0.1, image_size*0.03, image_size*0.03)][severity - 1]
    elif image_size < 224:  # Tiny-ImageNet size
        c = [(image_size*0, image_size*0, image_size*0.08),
             (image_size*0.05, image_size*0.3, image_size*0.06),
             (image_size*0.1, image_size*0.08, image_size*0.06),
             (image_size*0.1, image_size*0.03, image_size*0.03),
             (image_size*0.16, image_size*0.03, image_size*0.02)][severity - 1]
    else:  # ImageNet size
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but keeping for compatibility
             (244 * 2, 244 * 0.08, 244 * 0.2),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.07, 244 * 0.01, 244 * 0.02),
             (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// End Distortions ///////////////

def load_images_from_directory(dir_path):
    """Return the sorted paths of all images under ``dir_path`` (recursive)."""
    image_files = []
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm']
    
    if not os.path.exists(dir_path):
        print(f"Error: Directory {dir_path} does not exist")
        return []
        
    for root, _, files in os.walk(dir_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)

def validate_image_size(image_path):
    """Return the (square) side length of an image, or None if unreadable."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width != height:
                print(f"Warning: Image {image_path} is not square ({width}x{height}). Using min dimension.")
                return min(width, height)
            return width
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def get_corruption_functions():
    """The 15 ImageNet-C corruptions used in the paper."""
    d = collections.OrderedDict()
    d['Gaussian Noise'] = gaussian_noise
    d['Shot Noise'] = shot_noise
    d['Impulse Noise'] = impulse_noise
    d['Defocus Blur'] = defocus_blur
    d['Glass Blur'] = glass_blur
    d['Motion Blur'] = motion_blur
    d['Zoom Blur'] = zoom_blur
    d['Snow'] = snow
    d['Frost'] = frost
    d['Fog'] = fog
    d['Brightness'] = brightness
    d['Contrast'] = contrast
    d['Elastic'] = elastic_transform
    d['Pixelate'] = pixelate
    d['JPEG'] = jpeg_compression
    
    return d

def process_one_image(args):
    """Worker: load & preprocess image once, then apply all corruptions & severities."""
    (image_path, image_size, methods_items, severities, output_dir, preserve_structure, input_dir, print_prep) = args
    try:
        img = Image.open(image_path).convert('RGB')
        filename = os.path.basename(image_path)
        # relative path (preserve directory structure if requested)
        if preserve_structure:
            rel_path = os.path.relpath(os.path.dirname(image_path), input_dir)
            if rel_path == '.' or rel_path.startswith('..'):
                rel_path = ''
        else:
            rel_path = ''
        # preprocess once
        if image_size is not None:
            if image_size == 224:
                target_crop = 224
                resize_short = 256
                w, h = img.size
                if w <= h:
                    new_w = resize_short
                    new_h = int(round(h * resize_short / w))
                else:
                    new_h = resize_short
                    new_w = int(round(w * resize_short / h))
                if new_w <= 0 or new_h <= 0:
                    new_w = new_h = 256
                if print_prep:
                    print(f"[PREP 224] first example ({filename}): orig={w}x{h} -> resize_keep_ratio {new_w}x{new_h} -> center_crop {target_crop}x{target_crop}")
                img = img.resize((new_w, new_h), Image.BILINEAR)
                left = (new_w - target_crop) // 2
                top = (new_h - target_crop) // 2
                img = img.crop((left, top, left + target_crop, top + target_crop))
            else:
                ow, oh = img.size
                img = img.resize((image_size, image_size), Image.BILINEAR)
                if print_prep:
                    print(f"[PREP {image_size}] first example ({filename}): orig={ow}x{oh} -> direct_resize {image_size}x{image_size}")
        # apply corruptions
        for method_name, method in methods_items:
            for severity in severities:
                base_dir = os.path.join(output_dir, method.__name__, str(severity))
                save_dir = base_dir if not rel_path else os.path.join(base_dir, rel_path)
                if rel_path:
                    os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, filename)
                if os.path.exists(save_path):
                    continue
                try:
                    if method.__name__ not in ('pixelate', 'jpeg_compression'):
                        corrupted = method(img, severity, image_size)
                        Image.fromarray(np.uint8(corrupted)).save(save_path, quality=85)
                    else:
                        corrupted = method(img, severity, image_size)
                        corrupted.save(save_path, quality=85)
                except AttributeError as ae:
                    if "'Image' object has no attribute 'shape'" in str(ae):
                        corrupted = method(np.array(img), severity, image_size)
                        Image.fromarray(np.uint8(corrupted)).save(save_path, quality=85)
                    else:
                        raise ae
                except Exception as e:
                    print(f"Error processing {image_path} with {method.__name__} severity {severity}: {e}")
        return 1
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0


def process_images(input_dir, output_dir, image_size=None, methods=None, severities=None, preserve_structure=False, num_workers=1):
    """Apply every corruption/severity to each image (each image is loaded and preprocessed once)."""
    image_files = load_images_from_directory(input_dir)
    if not image_files:
        print(f"Error: No image files found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(image_files)} images to process")
    if image_size is None:
        first_image_size = None
        for img_path in image_files:
            first_image_size = validate_image_size(img_path)
            if first_image_size is not None:
                break
        if first_image_size is None:
            print("Error: Cannot detect image size from any image in the directory")
            sys.exit(1)
        image_size = first_image_size
        print(f"Detected image size: {image_size}x{image_size}")
    else:
        print(f"Using specified output image size: {image_size}x{image_size}")
    if image_size < 32:
        print(f"Error: Image size {image_size} is too small. Minimum supported size is 32.")
        sys.exit(1)
    d = get_corruption_functions()
    if methods is not None:
        filtered_d = collections.OrderedDict()
        for method_name in methods:
            if method_name in d:
                filtered_d[method_name] = d[method_name]
            else:
                print(f"Warning: Method {method_name} not found. Skipping.")
        d = filtered_d
    if severities is None:
        severities = list(range(1, 6))
    else:
        severities = [s for s in severities if 1 <= s <= 5]
        if not severities:
            print("Error: No valid severities provided (should be between 1 and 5).")
            sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    # pre-create directories to avoid race
    for method_name, method in d.items():
        for severity in severities:
            os.makedirs(os.path.join(output_dir, method.__name__, str(severity)), exist_ok=True)
    methods_items = list(d.items())
    # Prepare tasks
    tasks = []
    # first image will print preprocessing log
    if image_files:
        first = image_files[0]
        tasks.append((first, image_size, methods_items, severities, output_dir, preserve_structure, input_dir, True))
        for img_path in image_files[1:]:
            tasks.append((img_path, image_size, methods_items, severities, output_dir, preserve_structure, input_dir, False))
    total = len(tasks)
    processed = 0
    if num_workers > 1:
        print(f"Using multiprocessing with {num_workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_one_image, t) for t in tasks]
            for f in tqdm(concurrent.futures.as_completed(futures), total=total, desc='Total images'):
                processed += f.result()
    else:
        for t in tqdm(tasks, total=total, desc='Total images'):
            processed += process_one_image(t)
    print(f"Completed processing {processed} images (each with {len(methods_items)} corruptions * {len(severities)} severities)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply corruptions to images')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save corrupted images')
    parser.add_argument('--output_img_size', type=int, default=None, 
                        help='Size to resize images before applying corruptions (square images)')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes for parallel processing')
    parser.add_argument('--severities', type=int, nargs='+', default=[1, 2, 3, 4, 5], 
                        help='List of severity levels to generate (1-5). Example: --severities 3 5')
    parser.add_argument('--preserve_structure', action='store_true', help='Preserve original subdirectory structure in output')
    args = parser.parse_args()
    
    process_images(
        input_dir=args.input_dir,
        output_dir=f"{args.output_dir}",
        image_size=args.output_img_size,
        num_workers=args.num_workers,
        severities=args.severities,
        preserve_structure=args.preserve_structure
    )