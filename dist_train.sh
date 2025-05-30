CONFIG=$1
GPUS=$2
PORT=${PORT:-29515}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

    CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}  \
        --gpus $GPUS --seed 2023 