CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29516}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
	    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
		
