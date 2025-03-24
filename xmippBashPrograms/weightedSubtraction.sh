# Check if three arguments are provided
if [ $# -ne 3 ]; then
  echo "Usage: $0 inputReferenceMap inputSubtractionMap samplingRate"
  exit 1
fi

# Assign arguments to variables
LIG_MAP=$1
APO_MAP=$2
SAMPLING_RATE=$3

# Output variable for adjusted and difference map
DIR=$(dirname "$APO_MAP")
FILENAME=$(basename "$APO_MAP" .mrc)
EXTENSION="${APO_MAP##*.}"
APO_MAP_ADJ="${DIR}/${FILENAME}_adjusted.${EXTENSION}"

# Perform volume subtraction
echo "xmipp_volume_subtraction --i1 $LIG_MAP:mrc --i2 $APO_MAP:mrc -o $APO_MAP_ADJ --iter 5 --lambda 1.0 --cutFreq $SAMPLING_RATE --sigma 3 --radavg --computeEnergy"
xmipp_volume_subtraction --i1 $LIG_MAP:mrc --i2 $APO_MAP:mrc -o $APO_MAP_ADJ --iter 5 --lambda 1.0 --cutFreq $SAMPLING_RATE --sigma 3 --radavg --computeEnergy

# Iterate FACTOR from 0.1 to 0.9 in steps of 0.1
for FACTOR in $(seq 0.1 0.1 0.9); do
  APO_MAP_ADJ_MULT="${DIR}/${FILENAME}_adjusted_mult${FACTOR}.${EXTENSION}"
  DIFF_MAP="${DIR}/${FILENAME}_diff_${FACTOR}.${EXTENSION}"
  DIFF_MAP_MULT="${DIR}/${FILENAME}_diff_mult${FACTOR}.${EXTENSION}"

  echo "--- Operating maps for factor: $FACTOR"

  echo "xmipp_image_operate -i $APO_MAP_ADJ:mrc --mult $FACTOR -o $APO_MAP_ADJ_MULT"
  xmipp_image_operate -i $APO_MAP_ADJ:mrc --mult $FACTOR -o $APO_MAP_ADJ_MULT

  echo "xmipp_image_operate -i $LIG_MAP:mrc --minus $APO_MAP_ADJ_MULT:mrc -o $DIFF_MAP"
  xmipp_image_operate -i $LIG_MAP:mrc --minus $APO_MAP_ADJ_MULT:mrc -o $DIFF_MAP

  echo "xmipp_image_operate -i $DIFF_MAP:mrc --divide $FACTOR -o $DIFF_MAP_MULT"
  xmipp_image_operate -i $DIFF_MAP:mrc --divide $FACTOR -o $DIFF_MAP_MULT
done