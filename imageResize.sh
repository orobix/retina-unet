## img resize (3 channel)
for name in /home/moriaty/data/Projects/RetinalUnet/NEW/test/1st_manual/03_manual1.gif; do
  convert -resize 565x565! $name $name
done

## change file type
#mogrify -format tif *.png