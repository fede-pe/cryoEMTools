for FILE in *.mrc;
do DEFOCUS=$(shuf -i 4000-35000 -n1)
echo "xmipp_transform_filter -i $FILE -o "${FILE%.mrc}_ctf.mrc" --fourier ctfdef 300.000000 2.700000 0.070000 $DEFOCUS --sampling 1.000000 -v 0";
xmipp_transform_filter -i $FILE -o "${FILE%.mrc}_ctf.mrc" --fourier ctfdef 300.000000 2.700000 0.070000 $DEFOCUS --sampling 1.000000 -v 0;
done;
