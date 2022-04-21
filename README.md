## Circle Painter

This program uses a background image to recreate a new target image.
A binary mask must also be passed to tell the program which parts to recreate.


| <img src="imgs/vini.jpeg" alt="source" width="400"/> | <img src="imgs/vini_mask.jpg" alt="mask" width="400"/> |
|-|-|
| <img src="imgs/galaxy_4k.jpg" alt="background" width="400"/> | <img src="out.jpg" alt="out" width="400"/> |

### How it works
The program selects pixels from the target image and then calculates the
average color of different regions from the background image. The region
with the average color closest to the selected pixel value is then placed
where the pixel is on the image.


### Running
#### C++
First compile the script
```
./compile.sh
```

Then run
```
./main -t target_path -m mask_path -b background_path -a alpha
```

Where 
- target_path is where the image to be painted is;
- mask_path is where the mask to what parts of the image to paint is;
- background_path is the path of the image to use to reconstruct the target;
- alpha is the maximum alpha (between 0 and 1);


#### Python
Run `main.py`. Edit the following variables at the bottom of the file

```python
mask_path = "imgs/john_mask.jpg"
target_path = "imgs/john.jpg"
background_path = "imgs/space.jpg"
```

To point to the files you want

### Dependencies
The C++ program uses `opencv` and `openmp`. It is the recommended way to run,
as it is much faster than the python implementation, specially on bigger images.

The python program uses `skimage`, `numpy` and `nptyping`.