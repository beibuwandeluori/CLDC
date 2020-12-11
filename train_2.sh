BATCH_SIZE=12
DEVICE_ID=0
INPUT_SIZE=512
python train_ns.py --model_name=tf_efficientnet_b6_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=0
python train_ns.py --model_name=tf_efficientnet_b6_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=1
python train_ns.py --model_name=tf_efficientnet_b6_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=2
python train_ns.py --model_name=tf_efficientnet_b6_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=3
python train_ns.py --model_name=tf_efficientnet_b6_ns --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=4