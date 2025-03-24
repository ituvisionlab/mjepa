#%%
import torchio as tio
t1 = tio.datasets.Colin27().t1
print(t1.shape)
c, w, h, d = t1.shape
crop_pad = tio.CropOrPad((int(w/2), int(h/2), d))
t1_pad_crop = crop_pad(t1)
print(t1_pad_crop.shape)
subject = tio.Subject(t1=t1, crop_pad=t1_pad_crop)
subject.plot()
# %%
