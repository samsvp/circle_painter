#%%
import time
import numpy as np
import skimage # type: ignore
import skimage.transform as T # type: ignore
import matplotlib.pyplot as plt # type: ignore
from copy import deepcopy


from typing import Any, Callable, List, Tuple
from nptyping import NDArray, Shape, Int


def shift(img: np.ndarray, s: Tuple[int, int], 
        axis: Tuple[int, int]=(0,1)) -> np.ndarray:
    """Same as np.roll, but it fills with 0 the array"""
    shifted = np.roll(img, s, axis=axis)

    if (sy:=s[0]) < 0:
        shifted[sy:,:] = 0
    else:
        shifted[:sy,:] = 0
    if (sx:=s[1]) < 0:
        shifted[:,sx:] = 0
    else:
        shifted[:,:sx] = 0

    return shifted


def circle_mask(img: NDArray[Shape[Any, Any], Int],
        cx: int, cy: int, r: int) \
            -> Tuple[NDArray[Shape[Any, Any], Int], 
                     NDArray[Shape[Any, Any], Int]]:
    """Returns the image masked by a circle and the
    mask that produces the image"""
    x = np.arange(img.shape[0]).reshape(1,-1)
    y = np.arange(img.shape[1]).reshape(-1,1)
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    circle = np.zeros(img.shape, dtype=img.dtype)
    circle[mask] = img[mask]
    return (circle, mask)


def img_mean(img: NDArray[Shape[Any, Any], Int]) -> np.ndarray:
    """Returns the average of one image"""
    return np.mean(img, axis=(0,1))


def get_targets(mask: np.ndarray, target: np.ndarray,
        condition: Callable[[np.ndarray, int, int], bool]) \
            -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Returns the pixels that a circle could be placed"""
    pixel_pos, target_pixels = zip(*[((x, y), target[y,x])
        for y in range(mask.shape[0])
            for x in range(mask.shape[1])
                if condition(mask, x, y)])
    
    target_pixels = np.array(target_pixels)
    return (pixel_pos, target_pixels)


def place_circles(pixels_pos: List[Tuple[int, int]],
        target_pixels: np.ndarray, start: int, end: int,
        step: int, r: List[int], sx: float, 
        iters: List[int], img: np.ndarray, 
        d_bg: np.ndarray) -> np.ndarray:
    """Places circles inside the img"""

    for k in range(start, end):
        cr = int(r[k] * sx)

        sub_mask_pos, means = zip(*[
            # need to apply mask to become equal to the c++ code
            ((x, y), img_mean(
                circle_mask(d_bg, x, y, cr)[0]) )
            for x in range(cr, d_bg.shape[1] - cr, step)
                for y in range(cr, d_bg.shape[0] - cr, step)
        ])
        means = np.array(means)

        for _ in range(iters[k]):
            idx = np.random.randint(target_pixels.shape[0])
            target_pixel = target_pixels[idx]
            pixel_pos = pixels_pos[idx]
            mask_pixel_pos = (0, 0)

            # gets the minimum average difference between 
            # parts of the background and the target pixel
            diff = np.abs(target_pixel - means).sum(axis=1)
            mask_pixel_pos = sub_mask_pos[np.argmin(diff)]

            x = mask_pixel_pos[0]
            y = mask_pixel_pos[1]

            # get the image from the location
            min_avg_img, mask = circle_mask(d_bg, x, y, cr)
            # shift it to the right place
            mask = shift(mask, 
                (pixel_pos[1]-y, pixel_pos[0]-x))
            min_avg_img = shift(min_avg_img, 
                (pixel_pos[1]-y, pixel_pos[0]-x))
            img[mask] = min_avg_img[mask]

        print(f"Finished k={k}")
    return img


def circle_img(target_path: str, background_path: str,
        mask_path: str, step: int=5):
    """Creates the target image using pieces of the 
    background path. The given mask tells the program
    which parts to focus on"""
    mask = skimage.io.imread(mask_path)
    target = skimage.io.imread(target_path)
    background = skimage.io.imread(background_path)

    # radius of circles to use
    r = [200, 100, 20, 10, 5]
    # how many circles to use
    iters = [20, 50, 500, 500, 500]

    sx = 1.0
    sy = 1.0

    d_mask = T.resize(mask, 
        (sy * mask.shape[0], sx * mask.shape[1]))
    d_target = T.resize(target, 
        (sy * target.shape[0], sx * target.shape[1]))
    d_background = T.resize(background, 
        (sy * background.shape[0], sx * background.shape[1]))
    
    img = deepcopy(d_background)

    # get possible pixels
    pixels_pos, target_pixels = get_targets(
        d_mask, d_target, 
        lambda mask, x, y: all(mask[y, x] != [0,0,0])
    )
    # recreate circles using images
    img = place_circles(pixels_pos, target_pixels, 0, len(r), 
        step, r, sx, iters, img, d_background)
    
    return img


mask_path = "imgs/john_mask.jpg"
target_path = "imgs/john.jpg"
background_path = "imgs/space.jpg"

s = time.time()
img = circle_img(
    target_path, background_path, mask_path)
print(time.time() - s)

skimage.io.imsave("pyout.png", img)

plt.imshow(img)
plt.show()

# %%
