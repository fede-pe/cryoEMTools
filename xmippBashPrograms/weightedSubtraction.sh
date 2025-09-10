# Check if three arguments are provided
if [ $# -ne 3 ]; then
  echo "Usage: $0 inputReferenceMap inputSubtractionMap samplingRate"
  exit 1
fi

# Assign arguments to variables
LIG_MAP=$1
APO_MAP=$2
SAMPLING_RATE=$3

# Function to extract the correct extension and adjust filename
get_extension_and_filename() {
  local filename="$1"
  local suffix="$2"
  if [[ "$filename" =~ _half[12]\.mrc$ ]]; then
    local base="${filename%_half[12].mrc}"
    local half_suffix="${filename##${base}}"
    echo "${base}${suffix}${half_suffix}"
  else
    echo "${filename%.mrc}${suffix}.mrc"
  fi
}

# Function to format the factor
format_factor() {
  local factor="$1"
  echo "${factor/./}"
}

# Output variable for adjusted and difference map
DIR=$(dirname "$APO_MAP")
FILENAME=$(basename "$APO_MAP" .mrc)
EXTENSION=".mrc"
APO_MAP_ADJ="${DIR}/$(get_extension_and_filename "$FILENAME$EXTENSION" "_adjusted")"

# Perform volume subtraction
echo "xmipp_volume_subtraction --i1 $LIG_MAP:mrc --i2 $APO_MAP:mrc -o $APO_MAP_ADJ --iter 5 --lambda 1.0 --cutFreq $SAMPLING_RATE --sigma 3 --radavg --computeEnergy"
xmipp_volume_subtraction --i1 $LIG_MAP:mrc --i2 $APO_MAP:mrc -o $APO_MAP_ADJ --iter 5 --lambda 1.0 --cutFreq $SAMPLING_RATE --sigma 3 --radavg --computeEnergy

# Iterate FACTOR from 0.1 to 0.9 in steps of 0.1
for FACTOR in $(seq 0.1 0.1 0.9); do
  FORMATTED_FACTOR=$(format_factor "$FACTOR")
  APO_MAP_ADJ_MULT="${DIR}/$(get_extension_and_filename "$FILENAME$EXTENSION" "_adjusted_mult${FORMATTED_FACTOR}")"
  DIFF_MAP="${DIR}/$(get_extension_and_filename "$FILENAME$EXTENSION" "_diff_${FORMATTED_FACTOR}")"
  DIFF_MAP_MULT="${DIR}/$(get_extension_and_filename "$FILENAME$EXTENSION" "_diff_mult${FORMATTED_FACTOR}")"

  echo "--- Operating maps for factor: $FACTOR"

  echo "xmipp_image_operate -i $APO_MAP_ADJ:mrc --mult $FACTOR -o $APO_MAP_ADJ_MULT"
  xmipp_image_operate -i $APO_MAP_ADJ:mrc --mult $FACTOR -o $APO_MAP_ADJ_MULT

  echo "xmipp_image_operate -i $LIG_MAP:mrc --minus $APO_MAP_ADJ_MULT:mrc -o $DIFF_MAP"
  xmipp_image_operate -i $LIG_MAP:mrc --minus $APO_MAP_ADJ_MULT:mrc -o $DIFF_MAP

  echo "xmipp_image_operate -i $DIFF_MAP:mrc --divide $FACTOR -o $DIFF_MAP_MULT"
  xmipp_image_operate -i $DIFF_MAP:mrc --divide $FACTOR -o $DIFF_MAP_MULT
done