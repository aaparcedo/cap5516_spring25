import os
import numpy as np
from tifffile import imread
import glob
from collections import defaultdict
from tqdm import tqdm

def analyze_nuinsseg_dataset(root_dir="NuInsSeg"):
    """
    Analyzes the NuInsSeg dataset to compute statistics on point counts in images.
    
    Args:
        root_dir (str): Root directory of the NuInsSeg dataset
    
    Returns:
        dict: Dictionary containing statistics
    """
    # Statistics storage
    stats = {
        'organ_stats': defaultdict(list),
        'all_point_counts': [],
        'file_count': 0,
        'summary': {}
    }
    
    # Find all organs (top-level folders under NuInsSeg)
    organs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"Found {len(organs)} organs: {organs}")
    
    # Process each organ
    for organ in organs:
        organ_path = os.path.join(root_dir, organ)
        
        # Look for label masks directory
        label_masks_path = os.path.join(organ_path, "label masks")
        if not os.path.exists(label_masks_path):
            print(f"No 'label masks' directory found for {organ}, skipping.")
            continue
        
        # Find all TIF files
        tif_files = glob.glob(os.path.join(label_masks_path, "*.tif"))
        
        print(f"Processing {len(tif_files)} mask files for {organ}...")
        
        # Process each mask file
        for tif_file in tqdm(tif_files):
            stats['file_count'] += 1
            
            # Read the mask file
            try:
                mask = imread(tif_file)
                
                # Count unique labels (excluding background 0)
                unique_labels = np.unique(mask)
                point_count = len(unique_labels) - (1 if 0 in unique_labels else 0)
                
                # Store the point count
                stats['all_point_counts'].append(point_count)
                stats['organ_stats'][organ].append(point_count)
                
            except Exception as e:
                print(f"Error processing {tif_file}: {e}")
    
    # Calculate overall statistics
    if stats['all_point_counts']:
        all_counts = np.array(stats['all_point_counts'])
        
        stats['summary'] = {
            'min': int(np.min(all_counts)),
            'max': int(np.max(all_counts)),
            'mean': float(np.mean(all_counts)),
            'median': float(np.median(all_counts)),
            'std': float(np.std(all_counts)),
            'total_files': stats['file_count'],
            'total_points': int(np.sum(all_counts))
        }
        
        # Calculate per-organ statistics
        stats['organ_summary'] = {}
        for organ, counts in stats['organ_stats'].items():
            organ_counts = np.array(counts)
            stats['organ_summary'][organ] = {
                'min': int(np.min(organ_counts)),
                'max': int(np.max(organ_counts)),
                'mean': float(np.mean(organ_counts)),
                'median': float(np.median(organ_counts)),
                'std': float(np.std(organ_counts)),
                'count': len(counts)
            }
    
    return stats

def main():
    # Analyze the dataset
    print("Analyzing NuInsSeg dataset...")
    stats = analyze_nuinsseg_dataset()
    
    # Print overall statistics
    print("\n=====================")
    print("OVERALL STATISTICS")
    print("=====================")
    for key, value in stats['summary'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Print per-organ statistics
    print("\n=====================")
    print("PER-ORGAN STATISTICS")
    print("=====================")
    
    for organ, organ_stats in stats['organ_summary'].items():
        print(f"\n{organ}:")
        for key, value in organ_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Print histogram-like representation using ASCII
    print("\n=====================")
    print("POINT COUNT DISTRIBUTION")
    print("=====================")
    
    all_counts = stats['all_point_counts']
    min_count = min(all_counts)
    max_count = max(all_counts)
    
    # Create bins for ASCII histogram
    bin_count = 10
    bin_size = (max_count - min_count) / bin_count
    bins = defaultdict(int)
    
    for count in all_counts:
        bin_index = min(bin_count - 1, int((count - min_count) / bin_size))
        bins[bin_index] += 1
    
    # Find the maximum bin height for scaling
    max_bin_height = max(bins.values())
    
    # Print ASCII histogram
    print(f"\nDistribution of point counts ({min_count} to {max_count}):")
    for i in range(bin_count):
        bin_start = min_count + i * bin_size
        bin_end = min_count + (i + 1) * bin_size
        count = bins[i]
        bar = "#" * int(50 * count / max_bin_height)
        print(f"{bin_start:.1f}-{bin_end:.1f}: {bar} ({count})")

if __name__ == "__main__":
    main()