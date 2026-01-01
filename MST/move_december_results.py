import os
import shutil
from datetime import datetime
from pathlib import Path

def parse_date_from_dirname(dirname):
    """
    Parse date from directory name format: YYYY-MM-DD_HH-MM-SS
    Returns datetime object or None if parsing fails
    """
    try:
        date_str = dirname.split('_')[0]  # Get YYYY-MM-DD part
        return datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, IndexError):
        return None

def move_december_results(source_base='requests', target_base='requests_mistral', cutoff_date='2025-12-10'):
    """
    Move results from December 10th, 2025 onwards from source to target folder
    maintaining the same directory structure.
    
    Args:
        source_base: Base directory containing the results (default: 'requests')
        target_base: Destination directory (default: 'requests_mistral')
        cutoff_date: Minimum date to move (default: '2025-12-10')
    """
    # Convert cutoff date to datetime
    cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d')
    
    # Get the script's directory
    script_dir = Path(__file__).parent
    source_path = script_dir / source_base
    target_path = script_dir / target_base
    
    if not source_path.exists():
        print(f"Source directory '{source_path}' does not exist!")
        return
    
    moved_count = 0
    skipped_count = 0
    
    # Iterate through pattern folders (e.g., "128_128", "256_2048")
    for pattern_folder in source_path.iterdir():
        if not pattern_folder.is_dir():
            continue
        
        # Check if folder name matches the pattern (number_number)
        folder_name = pattern_folder.name
        if '_' not in folder_name:
            continue
        
        parts = folder_name.split('_')
        if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        
        print(f"\nProcessing pattern folder: {folder_name}")
        
        # Iterate through date subdirectories
        for date_folder in pattern_folder.iterdir():
            if not date_folder.is_dir():
                continue
            
            # Parse the date from the folder name
            folder_date = parse_date_from_dirname(date_folder.name)
            
            if folder_date is None:
                print(f"  Skipping '{date_folder.name}' - invalid date format")
                skipped_count += 1
                continue
            
            # Check if date is on or after cutoff
            if folder_date >= cutoff:
                # Create target directory structure
                target_pattern_folder = target_path / folder_name
                target_date_folder = target_pattern_folder / date_folder.name
                
                # Create parent directories if they don't exist
                target_date_folder.parent.mkdir(parents=True, exist_ok=True)
                
                # Move the entire subdirectory
                try:
                    shutil.move(str(date_folder), str(target_date_folder))
                    print(f"  ✓ Moved: {date_folder.name} ({folder_date.strftime('%Y-%m-%d')})")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ Error moving {date_folder.name}: {e}")
            else:
                print(f"  Skipped: {date_folder.name} ({folder_date.strftime('%Y-%m-%d')}) - before cutoff")
                skipped_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Moved: {moved_count} directories")
    print(f"  Skipped: {skipped_count} directories")
    print(f"  Target location: {target_path}")
    print(f"{'='*60}")

if __name__ == '__main__':
    # Run the function
    move_december_results()
