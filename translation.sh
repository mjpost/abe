# For sorting/file reasons let's have
# llama > tower > nllb > m2m > rachel's models
# rachel 64k > rachel 32k
# put the smaller model first
# i.e., run
# bash translation.sh rachel-32k rachel-64k en-de test, bash translation.sh rachel-32k llama3 en-de test
# NOT
# bash translation.sh rachel-64k rachel-32k en-de test, bash translation.sh llama3 rachel-32k en-de test



MODEL_ONE=$1
MODEL_TWO=$2
LANGUAGE_PAIR=$3
TESTSET=$4

# Get input paths
INPUT_ONE=$(python get-model-input.py $MODEL_ONE $LANGUAGE_PAIR $TESTSET)
INPUT_TWO=$(python get-model-input.py $MODEL_TWO $LANGUAGE_PAIR $TESTSET)

echo "Running translations on $MODEL_ONE and $MODEL_TWO for $LANGUAGE_PAIR and $TESTSET"
echo "Using files $INPUT_ONE and $INPUT_TWO"


CLEAN_MODEL_ONE_NAME=$(echo $MODEL_ONE | cut -d'/' -f2- | tr '/' '-')
CLEAN_MODEL_TWO_NAME=$(echo $MODEL_TWO | cut -d'/' -f2- | tr '/' '-')


OUTPUT_DIR="translations/$TESTSET/$LANGUAGE_PAIR"
mkdir -p $OUTPUT_DIR

# line by line is in sentences
mkdir -p "$OUTPUT_DIR/sentences/"

# merged output is in targets (for scoring)
mkdir -p "$OUTPUT_DIR/targets/"

OUTPUT_FILE="$OUTPUT_DIR/sentences/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"


MODEL_ONE=$(echo $MODEL_ONE | cut -d'/' -f1-2)
MODEL_TWO=$(echo $MODEL_TWO | cut -d'/' -f1-2)
# Run everything at half precision starting with en-de. If we get really different results, we'll revert
paste $INPUT_ONE $INPUT_TWO \
    | python ensembling/ensemble.py -m $MODEL_ONE $MODEL_TWO -l 256 --half beam \
    > $OUTPUT_FILE

# Get the targets
paste input_data/$TESTSET.$LANGUAGE_PAIR.line-numbers $OUTPUT_FILE | python combine-by-line-number.py > "$OUTPUT_DIR/targets/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"