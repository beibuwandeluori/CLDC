BATCH_SIZE=32
DEVICE_ID=4
INPUT_SIZE=600
#python train.py --model_name=efficientnet-b1 --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=0
#python train.py --model_name=efficientnet-b1 --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=1
python train.py --model_name=efficientnet-b1 --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=2
python train.py --model_name=efficientnet-b1 --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=3
python train.py --model_name=efficientnet-b1 --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=4