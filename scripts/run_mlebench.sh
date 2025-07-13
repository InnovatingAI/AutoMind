export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

IMAGE_ID=$(docker build -q --no-cache --platform=linux/amd64 -t automind -f mle-bench/Dockerfile . --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR)
echo $IMAGE_ID
python mle-bench/run_agent.py --agent-id automind --agent-name automind  --agent-dir automind --agent-config configs/mlebench.yaml --competition-set mle-bench/experiments/splits/aptos2019-blindness-detection.txt --data-dir /path/to/data --gpu-device 0
docker rmi $IMAGE_ID