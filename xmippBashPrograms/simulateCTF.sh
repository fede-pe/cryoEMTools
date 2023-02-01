for FILE in *.mrc; do 
DEFOCUS_U=$(shuf -i 4000-35000 -n1);
ASTIG_DIFF=$(($DEFOCUS_U/10));
ASTIG_RAND=$(shuf -i 0-$ASTIG_DIFF -n1);

SIGN=$(shuf -e -n1 0 2);

if [[$SIGN -gt 1]]
then
DEFOCUS_V=$(($DEFOCUS_U + $ASTIG_RAND));
else
DEFOCUS_V=$(($DEFOCUS_U - $ASTIG_RAND));
fi

ASTIG_ANGLE=$(shuf -i 0-180 -n1);
echo "xmipp_transform_filter -i $FILE -o "${FILE%.mrc}_ctf.mrc" --fourier ctfdefastig 300.000000 2.700000 0.070000 $DEFOCUS_U $DEFOCUS_V $ASTIG_ANGLE --sampling 1.000000 -v 0";
xmipp_transform_filter -i $FILE -o "${FILE%.mrc}_ctf.mrc" --fourier ctfdefastig 300.000000 2.700000 0.070000 $DEFOCUS_U $DEFOCUS_V $ASTIG_ANGLE --sampling 1.000000 -v 0;
done;
