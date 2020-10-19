import glob
from PIL import Image

# filepaths
fp_in = "./gradcam++/frames*.png"
fp_out = "./gradcam++.gif"

img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=1, loop=0)
