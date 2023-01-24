for FILE in *.mrc;
do echo "xmipp_transform_filter -i $FILE -o "${FILE%.mrc}_ctf.mrc" --fourier ctfdef 300.000000 2.700000 0.070000 20000.000000 --sampling 1.000000 -v 0";
xmipp_transform_filter -i $FILE -o "${FILE%.mrc}_ctf.mrc" --fourier ctfdef 300.000000 2.700000 0.070000 20000.000000 --sampling 1.000000 -v 0;
done;
