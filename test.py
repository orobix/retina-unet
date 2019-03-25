from PIL import Image
im = Image.open("./DRIVE_png/test/1st_manual/01_manual1.gif")
print(im.getbands())