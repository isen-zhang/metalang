#!/bin/bash

# Define domains and corresponding num_tokens
declare -A domains_tokens=( ["delete"]="1000M" ["deduplicate"]="1000M" ["duplicate"]="1000M" )

# Temporary directories for storing intermediate files
temp_dir="temp_data"
combined_dir="combined_data"
mkdir -p ${combined_dir}

# Loop through each domain and generate the files
for domain in "${!domains_tokens[@]}"; do
  num_token="${domains_tokens[$domain]}"

  # Define the command for each domain
  cmd="python -m tests.syn_data \
    --num_token ${num_token} \
    --data_config configs/${domain}.json"

  echo "Running command for ${domain} with ${num_token} tokens:"
  echo $cmd
  eval $cmd

  # Move generated files to temporary folder, renaming to avoid conflicts

done

# Combine and shuffle train, valid, and test sets from all domains
for split in "train" "valid" "test"; do
  # Concatenate all files of the same split type
  combined_file="${combined_dir}/combined_${split}.csv"
  cat data/*_${split}.csv > ${combined_file}

  # Shuffle the combined file
  shuf ${combined_file} -o ${combined_file}

  echo "Generated and shuffled combined file: ${combined_file}"
done

# Clean up temporary files
echo "Data generation and combination completed."

