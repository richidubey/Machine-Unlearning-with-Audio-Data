#!/usr/bin/env python3
"""
Download AudioMNIST dataset from HuggingFace
This script downloads the AudioMNIST dataset and saves it locally.
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
import datasets
from huggingface_hub import login


def download_AudioMNIST(output_dir: str = "./data", 
                       split: str = "train",
                       cache_dir: str = None,
                       num_proc: int = 4):
    """
    Download AudioMNIST dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset
        split: Dataset split to download ('train', 'test', or 'all')
        cache_dir: Cache directory for HuggingFace datasets
        num_proc: Number of processes for parallel processing
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AudioMNIST Dataset Download")
    print("=" * 60)
    print(f"Output directory: {output_path.absolute()}")
    print(f"Split: {split}")
    
    # Set cache directory if provided
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory: {cache_path.absolute()}")
    
    try:
        # Note: AudioMNIST might require authentication
        # Uncomment the following line if you need to login
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            login(token=token)
        
        print("\nDownloading AudioMNIST dataset...")
        print("Note: This is a large dataset and may take significant time.")
        print("The dataset will be cached for future use.")
        datasets.config.DOWNLOADER_TIMEOUT = 300 
        
        # Load the dataset
        if split == "all":
            dataset = load_dataset(
                "gilkeyio/AudioMNIST",
                cache_dir=cache_dir,
                num_proc=1
            )
        else:
            dataset = load_dataset(
                "gilkeyio/AudioMNIST",
                split=split,
                cache_dir=cache_dir,
                num_proc=1
            )
        
        # Save dataset info
        print(f"\nDataset downloaded successfully!")
        if isinstance(dataset, dict):
            for split_name, split_data in dataset.items():
                print(f"  {split_name}: {len(split_data)} samples")
        else:
            print(f"  Total samples: {len(dataset)}")
        
        # Save to disk
        print(f"\nSaving dataset to {output_path}...")
        dataset.save_to_disk(str(output_path / "AudioMNIST"))
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
        print(f"\nDataset saved to: {output_path / 'AudioMNIST'}")
        print("\nTo load the dataset in your code:")
        print(f"  from datasets import load_from_disk")
        print(f"  dataset = load_from_disk('{output_path / 'AudioMNIST'}')")
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nIf authentication is required:")
        print("1. Create a HuggingFace account at https://huggingface.co/")
        print("2. Create a token at https://huggingface.co/settings/tokens")
        print("3. Set the token as an environment variable:")
        print("   export HUGGINGFACE_TOKEN='your_token_here'")
        print("4. Or login using: huggingface-cli login")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download AudioMNIST dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save the dataset (default: ./data)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "all"],
        help="Dataset split to download (default: all)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of processes for parallel processing (default: 4)"
    )
    
    args = parser.parse_args()
    
    download_AudioMNIST(
        output_dir=args.output_dir,
        split=args.split,
        cache_dir=args.cache_dir,
        num_proc=args.num_proc
    )


if __name__ == "__main__":
    main()
