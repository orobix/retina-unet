## img resize (3 channel)
for name in /home/moriaty/data/Projects/RetinalUnet/DRIVE_png/training/images/*.tif; do
  convert -resize 565x565x3! $name $name
done

## change file type
#mogrify -format png *.gif