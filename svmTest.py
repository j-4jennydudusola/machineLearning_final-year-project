import matplotlib
matplotlib.use('MacOSX')

from nilearn.image import index_img
from nilearn.input_data import NiftiMasker
from nilearn import plotting


from nilearn.image import smooth_img, mean_img
from nilearn import image

results_img = smooth_img(['/Users/jennydudusola/Downloads/data/swarfMP_BBCI_VPTAQ-0002-*.img'],5)


epi_img = index_img(results_img[0], slice(0,100))
masker = NiftiMasker(mask_strategy='epi')
masker.fit(epi_img)

#extract time series data
trended = NiftiMasker(mask_strategy='epi')
detrended = NiftiMasker(mask_strategy='epi', detrend=True)
trended_data = trended.fit_transform(epi_img)
detrended_data = detrended.fit_transform(epi_img)


import matplotlib.pyplot as plt

# masked_data shape is (timepoints, voxels). We can plot the first 150
# timepoints from two voxels
plt.figure(figsize=(7,5))
plt.plot(trended_data[:150, :2])
plt.xlabel('Time [TRs]', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.xlim(0, 120)
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)
plt.show()

from sklearn import svm
clf = svm.SVC(kernel='linear')
#clf.fit(trended_data,detrended_data)

