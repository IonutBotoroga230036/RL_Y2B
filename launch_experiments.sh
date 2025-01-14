      
#!/bin/bash

# Define the number of experiments you want to run
NUM_EXPERIMENTS=5  # Changed from 30 to 5 (or 10)

# Define the ranges of your hyperparameters
LEARNING_RATES="1e-5 1e-4 1e-3"
BATCH_SIZES="32 128 256"
HIDDEN_UNITS="32 64 128"
GAMMAS="0.9 0.95 0.99 0.995"
GAE_LAMBDAS="0.9 0.95 0.99 1.0"
CLIP_RANGES="0.1 0.2 0.3"
REWARD_DISTANCE_SCALES="50 100 200"
STEP_PENALTIES="-0.5 -1 -2"
BONUS_REWARDS="50 100 200"
TOTAL_TIMESTEPS="100000"

# Loop to launch all the experiments
for (( i=0; i<$NUM_EXPERIMENTS; i++ ))
do
  # Generate random hyperparameters
  git pull origin main
  LR=$(echo $LEARNING_RATES | awk '{print $('"$RANDOM"%'${#LEARNING_RATES[@]}+1)}')
  BS=$(echo $BATCH_SIZES | awk '{print $('"$RANDOM"%'${#BATCH_SIZES[@]}+1)}')
  HU=$(echo $HIDDEN_UNITS | awk '{print $('"$RANDOM"%'${#HIDDEN_UNITS[@]}+1)}')
  G=$(echo $GAMMAS | awk '{print $('"$RANDOM"%'${#GAMMAS[@]}+1)}')
  GL=$(echo $GAE_LAMBDAS | awk '{print $('"$RANDOM"%'${#GAE_LAMBDAS[@]}+1)}')
  CR=$(echo $CLIP_RANGES | awk '{print $('"$RANDOM"%'${#CLIP_RANGES[@]}+1)}')
  RDS=$(echo $REWARD_DISTANCE_SCALES | awk '{print $('"$RANDOM"%'${#REWARD_DISTANCE_SCALES[@]}+1)}')
  SP=$(echo $STEP_PENALTIES | awk '{print $('"$RANDOM"%'${#STEP_PENALTIES[@]}+1)}')
  BR=$(echo $BONUS_REWARDS | awk '{print $('"$RANDOM"%'${#BONUS_REWARDS[@]}+1)}')
  TOTAL=$(echo $TOTAL_TIMESTEPS | awk '{print $('"$RANDOM"%'${#TOTAL_TIMESTEPS[@]}+1)}')

  # Print values for debugging purposes
  echo "Launching experiment with LR=$LR, BS=$BS, HU=$HU, G=$G, GL=$GL, CR=$CR, RDS=$RDS, SP=$SP, BR=$BR, TOTAL=$TOTAL"
  # Run the script with random parameters
  python RL_Training.py --learning_rate $LR --batch_size $BS --hidden_units $HU --gamma $G --gae_lambda $GL --clip_range $CR --reward_distance_scale $RDS --step_penalty $SP --bonus_reward $BR --total_timesteps $TOTAL &
done
wait
echo "All experiments have finished"

    