生成mappo的图像
uv run packages/SUMO/sumo_rl/outputs/plot.py -f ./results/sumo/4x4grid/mappo/handpicked/0916/log_conn0_ep2 -output mappo

uv run packages/SUMO/sumo_rl/res/plot.py -f ./results/sumo/4x4grid/mappo/sumo/max_reward/log_conn0_ep
3 -output 3xllm