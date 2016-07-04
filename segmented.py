import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data,io
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from PIL import Image

#image = data.coins()[50:-50, 50:-50]
image = io.imread('wordtest.jpg',as_grey=True)

###################
thresh = threshold_otsu(image)
binary = image > thresh

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1, adjustable='box-forced')
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].hist(image.ravel(), bins=256)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('off')

plt.show()
###################


# remove artifacts connected to image border
cleared = binary.copy()
clear_border(cleared)


img = img_as_float(binary)
segments_fz = felzenszwalb(img, scale=100, sigma=0.3, min_size=50)

img = Image.fromarray(segments_fz, 'RGB')

print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True,subplot_kw={'adjustable': 'box-forced'})
fig.set_size_inches(8, 3, forward=True)
fig.tight_layout()

ax[0].imshow(mark_boundaries(img, segments_fz))
ax[0].set_title("Felzenszwalbs's method")
for a in ax:
    a.set_xticks(())
    a.set_yticks(())
plt.show()
