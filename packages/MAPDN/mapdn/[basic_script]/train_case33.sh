# source train_case33.sh 0 l1 reproduction

if [ ! -d $3 ]
then
  mkdir $3
fi


export CUDA_VISIBLE_DEVICES=1
nohup uv run train.py --alg coma --alias $1 --mode distributed --scenario case33_3min_final --voltage-barrier-type $2 --save-path $3 > reproduction/logs/coma_33_$1_$2.out &
export CUDA_VISIBLE_DEVICES=1
nohup uv run train.py --alg iddpg --alias $1 --mode distributed --scenario case33_3min_final --voltage-barrier-type $2 --save-path $3 > reproduction/logs/iddpg_33_$1_$2.out &
export CUDA_VISIBLE_DEVICES=1
nohup uv run train.py --alg ippo --alias $1 --mode distributed --scenario case33_3min_final --voltage-barrier-type $2 --save-path $3 > reproduction/logs/ippo_33_$1_$2.out &
export CUDA_VISIBLE_DEVICES=0
nohup uv run train.py --alg matd3 --alias $1 --mode distributed --scenario case33_3min_final --voltage-barrier-type $2 --save-path $3 > reproduction/logs/matd3_33_$1_$2.out &
export CUDA_VISIBLE_DEVICES=0
nohup uv run train.py --alg maddpg --alias $1 --mode distributed --scenario case33_3min_final --voltage-barrier-type $2 --save-path $3 > reproduction/logs/maddpg_33_$1_$2.out &
export CUDA_VISIBLE_DEVICES=0
nohup uv run train.py --alg mappo --alias $1 --mode distributed --scenario case33_3min_final --voltage-barrier-type $2 --save-path $3 > reproduction/logs/mappo_33_$1_$2.out &
# export CUDA_VISIBLE_DEVICES=0
# nohup uv run train.py --alg maac --alias $1 --mode distributed --scenario case33_3min_final --voltage-barrier-type $2 --save-path $3 > reproduction/logs/maac_33_$1_$2.out &
export CUDA_VISIBLE_DEVICES=0
nohup uv run train.py --alg sqddpg --alias $1 --mode distributed --scenario case33_3min_final --voltage-barrier-type $2 --save-path $3 > reproduction/logs/sqddpg_33_$1_$2.out &