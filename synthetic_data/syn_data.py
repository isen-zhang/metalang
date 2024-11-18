import json
import pandas as pd
import numpy as np
import click
from seqthetic.domains import Deduplicate, Delete, Duplicate, Identity
from seqthetic.range import Range
from seqthetic.dataset_spec import DatasetSpec, SplitSpec, AddLossMask, OffsetVocabulary
from seqthetic.vocabulary import Vocabulary

def load_dataset_spec(data_config, num_token):
    # Initialize the domain based on the data_config name
    if data_config['name'] == 'delete':
        domain = Delete(
            sequence_length=Range(min=data_config['sequence_length_min'], max=data_config['sequence_length_max']),
            should_delete_ratio=data_config['should_delete_ratio'],
            seq_to_delete_length=Range(min=data_config['seq_to_delete_length_min'], max=data_config['seq_to_delete_length_max']),
            vocab=Vocabulary(num_vocab=data_config['vocab_size']),
        )
    elif data_config['name'] == 'deduplicate':
        domain = Deduplicate(
            sequence_length=Range(min=data_config['sequence_length_min'], max=data_config['sequence_length_max']),
            unique_token_ratio=Range(min=data_config['unique_token_ratio_min'], max=data_config['unique_token_ratio_max']),
            repeat_token_ratio=Range(min=data_config['repeat_token_ratio_min'], max=data_config['repeat_token_ratio_max']),
            shuffle_ratio=Range(min=data_config['shuffle_ratio_min'], max=data_config['shuffle_ratio_max']),
            vocab=Vocabulary(num_vocab=data_config['vocab_size']),
        )
    elif data_config['name'] == 'duplicate':
        domain = Duplicate(
            sequence_length=Range(min=data_config['sequence_length_min'], max=data_config['sequence_length_max']),
            vocab=Vocabulary(num_vocab=data_config['vocab_size']),
        )
    elif data_config['name'] == 'identity' or data_config['name'] == 'reverse':
        domain = Identity(
            sequence_length=Range(min=data_config['sequence_length_min'], max=data_config['sequence_length_max']),
            vocab=Vocabulary(num_vocab=data_config['vocab_size']),
        )
    else:
        raise ValueError(f"Unsupported domain type: {data_config['name']}")

    # Define postprocessors including AddLossMask with mask_delimiter setting

    postprocessors = [
        AddLossMask(mask_delimiter=True),  # Add promoter to the mask
        OffsetVocabulary()            
    ]

    # Create and return DatasetSpec object with split configuration
    split_spec = SplitSpec(**data_config["split"])

    return DatasetSpec(
        name=data_config['name'],
        num_token=num_token,
        vocab_size=data_config['vocab_size'],
        domains=[domain],
        postprocessors=postprocessors,
        split=split_spec,
    )

def generate_promoter_sequence(vocab_size, promoter_length):
    """Generate a promoter sequence of specified length within a given vocab size."""
    return np.random.randint(0, vocab_size, size=promoter_length)

def get_promoter_length(domain_name, promoter_length_range):
    """Generate a consistent promoter length based on the domain name."""
    # Use the hash of the domain name to seed the random generator
    seed = hash(domain_name) % (2**32)  # Ensures compatibility with np.random.seed
    rng = np.random.default_rng(seed)
    
    # Generate the promoter length within the specified range
    promoter_length = rng.integers(promoter_length_range[0], promoter_length_range[1] + 1)
    return promoter_length

@click.command()
@click.option('--num_token', default='500M', help='Number of tokens')
@click.option('--data_config', type=str, help='Path to data config JSON file')
def main(num_token: str, data_config: str):
    # Load the configuration file
    with open(data_config) as f:
        data_config = json.load(f)

    # Initialize DatasetSpec based on configuration
    spec = load_dataset_spec(data_config, num_token)
    dataset_result = spec.make_dataset()
    promoter_length_range = data_config.get("promoter_length_range", (5, 10))
    promoter_length = np.random.randint(promoter_length_range[0], promoter_length_range[1] + 1)
    
    # Flatten domain_sequences and corresponding original loss_masks
    all_sequences = []
    all_combined_masks = []

    # Generate promoter and prepend it to each sequence in each domain
    for domain, sequences, domain_column in zip(dataset_result.domains, dataset_result.domain_sequences, dataset_result.domain_columns):
        promoter_length = get_promoter_length(data_config['name'], promoter_length_range)
        promoter_sequence = generate_promoter_sequence(data_config['vocab_size'], promoter_length)


        original_masks = domain_column["loss_mask"]
        for sequence, original_mask in zip(sequences, original_masks):
            # Combine promoter with sequence
            full_sequence = np.concatenate([promoter_sequence, sequence])
            all_sequences.append(full_sequence)
            
            # Create the promoter mask with 1s for the promoter length
            promoter_mask = np.ones(promoter_length, dtype=int)
            
            # Combine promoter mask with the original mask for the sequence
            combined_mask = np.concatenate([promoter_mask, original_mask])
            all_combined_masks.append(combined_mask)
  

    # Shuffle dataset if required
    if spec.split.shuffle_dataset:
        combined = list(zip(all_sequences, all_combined_masks))
        np.random.shuffle(combined)
        all_sequences, all_combined_masks = zip(*combined)

    # Get indices for train, validation, and test splits
    train_idx, val_idx = spec.split.get_index(len(all_sequences))
    
    # Split data according to the indices
    train_sequences = all_sequences[:train_idx]
    train_masks = all_combined_masks[:train_idx]

    valid_sequences = all_sequences[train_idx:val_idx]
    valid_masks = all_combined_masks[train_idx:val_idx]

    test_sequences = all_sequences[val_idx:]
    test_masks = all_combined_masks[val_idx:]

    # Create DataFrames for each split and save to CSV
    for split_name, split_sequences, split_masks in zip(
        ["train", "valid", "test"],
        [train_sequences, valid_sequences, test_sequences],
        [train_masks, valid_masks]
    ):
        df = pd.DataFrame({
            "dependency": [0 for _ in range(len(split_sequences))],
            "sequence": split_sequences,
            "loss_mask": split_masks
        })
        df["sequence"] = df["sequence"].apply(lambda x: x.tolist())
        df["loss_mask"] = df["loss_mask"].apply(lambda x: x.tolist())
        df.to_csv(f"data/{data_config['name']}_{num_token}_{split_name}.csv", index=False)

if __name__ == '__main__':
    main()
