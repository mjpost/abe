v=$1;
echo "Running translations on baseline_en-de_"$v"k_ep1"
MODEL_ONE="rewicks/baseline_en-de_"$v"k_ep1"
for ep in {2,3,4,5,10,15,20,25}; do
    MODEL_TWO="rewicks/baseline_en-de_"$v"k_ep"$ep
    CLEAN_MODEL_ONE_NAME=$(echo $MODEL_ONE | cut -d'/' -f2)
    CLEAN_MODEL_TWO_NAME=$(echo $MODEL_TWO | cut -d'/' -f2)
    OUTPUT="outputs/sentences/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"
    cat ../../wmt24.en-de.en.sentences | cut -f2 | python -u interpolate-translate.py -m $MODEL_ONE -m $MODEL_TWO > $OUTPUT
done

echo "Running translations on baseline_en-de_"$v"k_ep2"
MODEL_ONE="rewicks/baseline_en-de_"$v"k_ep2"
for ep in {3,4,5,10,15,20,25}; do
    MODEL_TWO="rewicks/baseline_en-de_"$v"k_ep"$ep
    CLEAN_MODEL_ONE_NAME=$(echo $MODEL_ONE | cut -d'/' -f2)
    CLEAN_MODEL_TWO_NAME=$(echo $MODEL_TWO | cut -d'/' -f2)
    OUTPUT="outputs/sentences/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"
    cat ../../wmt24.en-de.en.sentences | cut -f2 | python -u interpolate-translate.py -m $MODEL_ONE -m $MODEL_TWO > $OUTPUT
done

echo "Running translations on baseline_en-de_"$v"k_ep3"
MODEL_ONE="rewicks/baseline_en-de_"$v"k_ep3"
for ep in {4,5,10,15,20,25}; do
    MODEL_TWO="rewicks/baseline_en-de_"$v"k_ep"$ep
    CLEAN_MODEL_ONE_NAME=$(echo $MODEL_ONE | cut -d'/' -f2)
    CLEAN_MODEL_TWO_NAME=$(echo $MODEL_TWO | cut -d'/' -f2)
    OUTPUT="outputs/sentences/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"
    cat ../../wmt24.en-de.en.sentences | cut -f2 | python -u interpolate-translate.py -m $MODEL_ONE -m $MODEL_TWO > $OUTPUT
done

echo "Running translations on baseline_en-de_"$v"k_ep4"
MODEL_ONE="rewicks/baseline_en-de_"$v"k_ep4"
for ep in {5,10,15,20,25}; do
    MODEL_TWO="rewicks/baseline_en-de_"$v"k_ep"$ep
    CLEAN_MODEL_ONE_NAME=$(echo $MODEL_ONE | cut -d'/' -f2)
    CLEAN_MODEL_TWO_NAME=$(echo $MODEL_TWO | cut -d'/' -f2)
    OUTPUT="outputs/sentences/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"
    cat ../../wmt24.en-de.en.sentences | cut -f2 | python -u interpolate-translate.py -m $MODEL_ONE -m $MODEL_TWO > $OUTPUT
done

echo "Running translations on baseline_en-de_"$v"k_ep5"
MODEL_ONE="rewicks/baseline_en-de_"$v"k_ep5"
for ep in {10,15,20,25}; do
    MODEL_TWO="rewicks/baseline_en-de_"$v"k_ep"$ep
    CLEAN_MODEL_ONE_NAME=$(echo $MODEL_ONE | cut -d'/' -f2)
    CLEAN_MODEL_TWO_NAME=$(echo $MODEL_TWO | cut -d'/' -f2)
    OUTPUT="outputs/sentences/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"
    cat ../../wmt24.en-de.en.sentences | cut -f2 | python -u interpolate-translate.py -m $MODEL_ONE -m $MODEL_TWO > $OUTPUT
done

echo "Running translations on baseline_en-de_"$v"k_ep10"
MODEL_ONE="rewicks/baseline_en-de_"$v"k_ep10"
for ep in {15,20,25}; do
    MODEL_TWO="rewicks/baseline_en-de_"$v"k_ep"$ep
    CLEAN_MODEL_ONE_NAME=$(echo $MODEL_ONE | cut -d'/' -f2)
    CLEAN_MODEL_TWO_NAME=$(echo $MODEL_TWO | cut -d'/' -f2)
    OUTPUT="outputs/sentences/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"
    cat ../../wmt24.en-de.en.sentences | cut -f2 | python -u interpolate-translate.py -m $MODEL_ONE -m $MODEL_TWO > $OUTPUT
done

echo "Running translations on baseline_en-de_"$v"k_ep15"
MODEL_ONE="rewicks/baseline_en-de_"$v"k_ep15"
for ep in {20,25}; do
    MODEL_TWO="rewicks/baseline_en-de_"$v"k_ep"$ep
    CLEAN_MODEL_ONE_NAME=$(echo $MODEL_ONE | cut -d'/' -f2)
    CLEAN_MODEL_TWO_NAME=$(echo $MODEL_TWO | cut -d'/' -f2)
    OUTPUT="outputs/sentences/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"
    cat ../../wmt24.en-de.en.sentences | cut -f2 | python -u interpolate-translate.py -m $MODEL_ONE -m $MODEL_TWO > $OUTPUT
done

echo "Running translations on baseline_en-de_"$v"k_ep20"
MODEL_ONE="rewicks/baseline_en-de_"$v"k_ep20"
ep="25";
MODEL_TWO="rewicks/baseline_en-de_"$v"k_ep"$ep
CLEAN_MODEL_ONE_NAME=$(echo $MODEL_ONE | cut -d'/' -f2)
CLEAN_MODEL_TWO_NAME=$(echo $MODEL_TWO | cut -d'/' -f2)
OUTPUT="outputs/sentences/$CLEAN_MODEL_ONE_NAME+$CLEAN_MODEL_TWO_NAME"
cat ../../wmt24.en-de.en.sentences | cut -f2 | python -u interpolate-translate.py -m $MODEL_ONE -m $MODEL_TWO > $OUTPUT