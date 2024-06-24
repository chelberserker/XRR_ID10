from XRR import XRR
from XRR import rebin
import numpy as np
import matplotlib.pyplot as plt

FileDir = 'E:/Bersenev/ID10_ID10_Apr24/RAW_DATA/A9_84_4000/A9_84_4000_0001/'
FileName = '\A9_84_4000_0001.h5'
ScanN_list = [27]
file = FileDir+FileName

refl = XRR(file, ScanN_list, PX0=382, PY0 = 327, dPX=8, dPY=8, I0=1.10e13, transmission_name='autof_eh1_transm', att_name='autof_eh1_curratt', energy_name='monoe')
refl.correct_doubles()
refl.footprint_correction(2, 16, correct_dir_beam=True)
refl.produce_Qmap(SDD=830)

#3 detector images in one figure
fig, [ax1, ax2, ax3] = plt.subplots(3,1, sharex=True, figsize=(8,8), layout='tight')
refl.show_detector_image(20, ax=ax1, plot_cross=False)
ax1.set_xlim(100, 440)
ax1.set_yticks([250, 300, 350, 400])
ax1.text(260, 380, '$\\alpha_i = {:.3f}^o$'.format(refl.alpha_i[20]), color='r', fontsize=14)
ax1.set_title('')
ax1.set_xlabel('')
refl.show_detector_image(117, ax=ax2, plot_cross=False)
ax2.set_xlim(100, 440)
ax2.set_yticks([250, 300, 350, 400])
ax2.text(260, 380, '$\\alpha_i = {:.3f}^o$'.format(refl.alpha_i[117]), color='r', fontsize=14)
ax2.set_title('')
ax2.set_xlabel('')
refl.show_detector_image(162, ax=ax3, plot_cross=False)
ax3.text(260, 380, '$\\alpha_i = {:.3f}^o$'.format(refl.alpha_i[162]), color='r', fontsize=14)
ax3.set_xlim(100, 440)
ax3.set_yticks([250, 300, 350, 400])
ax3.set_title('')
fig.savefig('detector_frames.png', dpi=300)

#1 detector image and 1 profile
fig, [ax1, ax2] = plt.subplots(2,1, sharex=True, figsize=(8,8), layout='tight')
ax1.imshow(np.log10(refl.data[117]+1e-3))
ax1.text(260, 380, '$\\alpha_i = {:.3f}^o$'.format(refl.alpha_i[117]), color='r', fontsize=14)
ax1.annotate('Specular \nreflection',(refl.PX0, refl.PY0), (refl.PX0, refl.PY0+40), color='r', arrowprops=dict(arrowstyle='->', color='r'), fontsize=14)
ax1.annotate('Bragg peak',(350, refl.PY0), (300, refl.PY0-40), color='r', arrowprops=dict(arrowstyle='->', color='r'), fontsize=14)
ax1.set_xlim(100, 440)
ax1.set_ylim(240, 410)
ax1.set_yticks([250, 300, 350, 400])
ax1.set_ylabel('Detector pixel, Y')
profile = np.sum(refl.data[117][300:360], axis=0)
ax2.semilogy(profile[:428])
ax2.set_xlim(100, 440)

ax2.annotate('Specular \nreflection',(refl.PX0, profile[refl.PX0]), (refl.PX0-20, 1e2), color='k', arrowprops=dict(arrowstyle='->', color='k'), fontsize=14)
ax2.annotate('Bragg peak',(350-5, profile[350]), (250, 1e4), color='k', arrowprops=dict(arrowstyle='->', color='k'), fontsize=14)

ax2.set_xlabel('Detector pixel, X')
ax2.set_ylabel('Integrated intensity, cts')

plt.savefig('Annotated_detector_profile.png', dpi=300)