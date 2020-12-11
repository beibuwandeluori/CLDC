BATCH_SIZE=16
DEVICE_ID=4
INPUT_SIZE=512
python train_ns_soft.py --model_name=tf_efficientnet_b5_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=0
python train_ns_soft.py --model_name=tf_efficientnet_b5_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=1
python train_ns_soft.py --model_name=tf_efficientnet_b5_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=2
python train_ns_soft.py --model_name=tf_efficientnet_b5_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=3
python train_ns_soft.py --model_name=tf_efficientnet_b5_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=4