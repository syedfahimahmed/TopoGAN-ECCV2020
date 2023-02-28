THD=$1
TRAINING_FILE=("sample_A_20160501.hdf" "sample_B_20160501.hdf")
OUTPUT_NAME=("CREMI_AB_64_64_train_$THD" "CREMI_AB_64_64_test_$THD")
IODIR="C:\Users\nelsite\Desktop\Coding_with_Fahim\Topological_Segmentation\TopoSegNetSimple\TopoSegNetSimple\datagenerator"
N_RUNS=20
PVPYTHON="C:\Program Files\ParaView 5.11.0\bin\pvpython"

COMMON_FLAG="--scalar_size 64 64 --image_size 64 64"

for I in `seq $N_RUNS`; do
    RAND_IDX=$[ $RANDOM % 2 ]
    INPUT_DIR=$IODIR${TRAINING_FILE[$RAND_IDX]}
    O1_DIR=$IODIR${OUTPUT_NAME[0]}
    O2_DIR=$IODIR${OUTPUT_NAME[1]}
    $PVPYTHON hdf5converter.py $INPUT_DIR $O1_DIR 100 --index $RAND_IDX $COMMON_FLAG &>> $O1_DIR.log
    $PVPYTHON hdf5converter.py $INPUT_DIR $O2_DIR 20 --index $RAND_IDX $COMMON_FLAG &>> $O2_DIR.log

    echo "Thread $THD finished $I / $N_RUNS  on random_idx:$RAND_IDX"
done
