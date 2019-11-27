import matplotlib
matplotlib.use('MacOSX')

from nilearn.image import smooth_img
from nilearn import image

results_img = smooth_img(['~/PycharmProjects/project3/data/swarfMP_BBCI_VPTAQ-0002-*.img'],5)

from nilearn.image import index_img
from nilearn.input_data import NiftiMasker
from nilearn import plotting

epi_img = index_img(results_img[0], slice(0,100))
masker = NiftiMasker(mask_strategy='epi')
masker.fit(epi_img)

plotting.plot_roi(masker.mask_img_, image.mean_img(results_img), title='EPI automatic mask')
plotting.show()

plotting.plot_prob_atlas(epi_img, title='A Probabilistic Atlas of 4D Brain', colorbar='True')
plotting.show()






