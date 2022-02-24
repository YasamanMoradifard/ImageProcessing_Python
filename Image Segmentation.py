# Importing libraries
import skimage
import numpy as np
import matplotlib.pyplot as plt

# Importing data
from skimage import data
coins = data.coins()

# Data visualization
plt.imshow(coins, cmap='gray')

# Image denoising
from skimage import filters
# a median filter with size=(5,5)
coins_denoised = filters.median (coins, selem= np.ones((5,5)))

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,5))

ax0.set_title('original image')
ax0.imshow(coins, cmap = 'gray')

ax1.set_title('smoothed image')
ax1.imshow(coins_denoised, cmap = 'gray')

# Finding edges of objects
from skimage import feature
# The less the sigma, the more noises
edges = skimage.feature.canny(coins_denoised, sigma = 4)

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,5))
ax0.set_title('original image')
ax0.imshow(coins, cmap = 'gray')

ax1.set_title('edges of objets')
ax1.imshow(edges, cmap = 'gray')

# Converting images to a distance map
from scipy.ndimage import distance_transform_edt
# distance_transform_edt is a euclidian distance parameter
dt = distance_transform_edt (~edges)

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,5))
ax0.set_title('original image')
ax0.imshow(coins, cmap = 'gray')

ax1.set_title('edges of objets')
ax1.imshow(dt, cmap = 'gray')

# Finding peaks/features
local_max = feature.peak_local_max(dt, indices = False, min_distance = 5)

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,5))
ax0.set_title('original image')
ax0.imshow(coins, cmap = 'gray')

ax1.set_title('center of coins')
ax1.imshow(local_max, cmap = 'gray')

## actual positions of features
peak_idx = feature.peak_local_max(dt, indices = True, min_distance = 5)
peak_idx[:5]

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,5))
ax0.set_title('original image')
ax0.imshow(coins, cmap = 'gray')

ax1.set_title('center of coins')
plt.plot(peak_idx[:,1], peak_idx[:,0], 'r.')
ax1.imshow(dt, cmap = 'gray')

# Labeling features
from skimage import measure
# each dot gets a new integer label
makers = measure.label(local_max)

from skimage import morphology, segmentation
# inverting the distance map to a segmented image 
labels = morphology.watershed(-dt, makers)

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,5))
ax0.set_title('original image')
ax0.imshow(coins, cmap = 'gray')

ax1.set_title('segmented image')
ax1.imshow(segmentation.mark_boundaries(coins, labels), cmap = 'gray')

from skimage import color
f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,5))
ax0.set_title('original image')
ax0.imshow(coins, cmap = 'gray')

ax1.set_title('segmented coins')
ax1.imshow(color.label2rgb(labels, image=coins))

# Segmentation
regions = measure.regionprops(labels, intensity_image = coins)
region_means = [r.mean_intensity for r in regions]
plt.hist(region_means, bins=20)

from sklearn.cluster import KMeans
model = KMeans (n_clusters = 2)

region_means = np.array(region_means).reshape(-1,1)

model.fit(np.array(region_means).reshape(-1,1))
print(model.cluster_centers_)

# labels of forground and background
bg_fg_labels = model.predict(region_means)

classified_labels = labels.copy()
for bg_fg, region in zip(bg_fg_labels, regions):
    classified_labels[tuple(region.coords.T)] = bg_fg

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15,5))
ax0.set_title('original image')
ax0.imshow(coins, cmap = 'gray')

ax1.set_title('segmented image')
ax1.imshow(color.label2rgb(classified_labels, image = coins), cmap = 'gray')