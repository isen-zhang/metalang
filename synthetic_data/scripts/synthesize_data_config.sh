num_token="5M"
# domain="delete"
# domain="deduplicate"
domain="duplicate"

cmd="python -m tests.synthesize \
  --num_token ${num_token} \
  --data_config configs/${domain}.json"
echo $cmd
eval $cmd
