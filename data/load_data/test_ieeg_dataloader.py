import os
import sys
import torch
import numpy as np
import pickle
from torch_geometric.data import Data
import mne

# Add the project root to the path
sys.path.append('/dartfs-hpc/rc/home/w/f004r7w/ieeg_ml_project/TSADC')

from data.load_data.data_ieeg import IEEGDataset, IEEG_DataModule

TMP_LOC = "/dartfs-hpc/scratch/f004r7w/tmp"

MOCK_DATA_DIR = f"{TMP_LOC}/mock_ieeg_data"




def test_ieeg_dataset():
    """Test the IEEGDataset class"""
    print("Testing IEEGDataset...")
    
    # Test parameters
    raw_data_path = "/dartfs-hpc/scratch/f004r7w/data_files/ds003029-processed/preprocessed_data"  # Update this path
    preproc_save_dir = "/dartfs-hpc/scratch/f004r7w/data_files/ds003029-processed/tsadc_processed_data"
    seq_len = 2  # 2 seconds
    num_nodes = 1 # Adjust based on your data
    freq = 250  # Target frequency
    
    # Create test dataset
    try:
        dataset = IEEGDataset(
            root=preproc_save_dir,
            raw_data_path=raw_data_path,
            file_marker=None,  # Will scan directory
            split="test",
            seq_len=seq_len,
            num_nodes=num_nodes,
            adj_mat_dir=None,  # Will create identity matrix
            freq=freq,
            scaler=None,
        )
        
        print(f"Dataset created successfully!")
        print(f"Number of samples: {len(dataset)}")
        print(f"Labels distribution: {np.unique(dataset.labels, return_counts=True)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample data shape: {sample.x.shape}")
            print(f"Sample label: {sample.y}")
            print(f"Edge index shape: {sample.edge_index.shape}")
            print(f"Adjacency matrix shape: {sample.adj_mat.shape}")
            
            # Verify data properties
            expected_time_points = freq * seq_len
            assert sample.x.shape[1] == expected_time_points, f"Expected {expected_time_points} time points, got {sample.x.shape[1]}"
            assert sample.x.shape[2] == 1, f"Expected 1 feature dimension, got {sample.x.shape[2]}"
            print("✓ Data shape validation passed")
            
    except Exception as e:
        print(f"Error testing dataset: {e}")
        return False
    
    return True


def test_ieeg_datamodule():
    """Test the IEEG_DataModule class"""
    print("\nTesting IEEG_DataModule...")
    
    # Test parameters
    raw_data_path = "/dartfs-hpc/scratch/f004r7w/data_files/ds003029-processed/preprocessed_data"  # Update this path
    preproc_save_dir = "/dartfs-hpc/scratch/f004r7w/data_files/ds003029-processed/tsadc_processed_data"
    seq_len = 2
    num_nodes = 64
    train_batch_size = 4
    test_batch_size = 4
    num_workers = 0  # Use 0 for testing to avoid multiprocessing issues
    freq = 250
    
    try:
        datamodule = IEEG_DataModule(
            raw_data_path=raw_data_path,
            preproc_save_dir=preproc_save_dir,
            seq_len=seq_len,
            num_nodes=num_nodes,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            freq=freq,
            adj_mat_dir=None,
            standardize=True,
            balanced_sampling=False,
            pin_memory=False,
        )
        
        print("DataModule created successfully!")
        
        # Test dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        print(f"Train dataset size: {len(datamodule.train_dataset)}")
        print(f"Val dataset size: {len(datamodule.val_dataset)}")
        print(f"Test dataset size: {len(datamodule.test_dataset)}")
        
        # Test loading a batch
        if len(datamodule.train_dataset) > 0:
            for batch in train_loader:
                print(f"Batch x shape: {batch.x.shape}")
                print(f"Batch y shape: {batch.y.shape}")
                print(f"Batch edge_index shape: {batch.edge_index.shape}")
                print(f"Unique labels in batch: {torch.unique(batch.y)}")
                break
            
            print("✓ DataModule validation passed")
        
    except Exception as e:
        print(f"Error testing datamodule: {e}")
        return False
    
    return True


# def test_resampling_functionality():
#     """Test the resampling functionality with mock data"""
#     print("\nTesting resampling functionality with mock data...")
    
#     try:
#         # Create a mock dataset to test resampling
#         dataset = IEEGDataset(
#             root=f"{TMP_LOC}/test_resample",
#             raw_data_path=MOCK_DATA_DIR,
#             file_marker=None,
#             split="test",
#             seq_len=1,
#             num_nodes=10,
#             adj_mat_dir=None,
#             freq=250,  # Target frequency
#             scaler=None,
#         )
        
#         # Test resampling function with mock data
#         original_data = np.random.randn(10, 1000)  # 10 channels, 1000 time points
#         original_freq = 1000
#         target_freq = 250
        
#         resampled_data = dataset._resample_data(original_data, original_freq, target_freq)
        
#         expected_length = int(1000 * (target_freq / original_freq))
#         assert resampled_data.shape[1] == expected_length, f"Expected {expected_length} time points, got {resampled_data.shape[1]}"
#         assert resampled_data.shape[0] == 10, f"Expected 10 channels, got {resampled_data.shape[0]}"
        
#         print(f"Original data shape: {original_data.shape}")
#         print(f"Resampled data shape: {resampled_data.shape}")
#         print("✓ Resampling functionality validation passed")
        
#     except Exception as e:
#         print(f"Error testing resampling: {e}")
#         return False
    
#     return True


def create_mock_data(mock_data_dir):
    """Create mock iEEG data for testing"""
    print(f"\nCreating mock data in {mock_data_dir}...")
    
    os.makedirs(mock_data_dir, exist_ok=True)
    
    # Create patient directories with mock data
    for patient_id in ['patient_001', 'patient_002']:
        patient_dir = os.path.join(mock_data_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        
        for label, label_name in [(0, 'preictal'), (1, 'ictal'), (2, 'postictal')]:
            for file_idx in range(2):  # 2 files per label
                # Create mock iEEG data (10 channels, 5000 time points at 1000 Hz = 5 seconds)
                mock_data = np.random.randn(10, 5000)
                
                # Create MNE Raw object
                sfreq = 1000  # Sampling frequency
                ch_names = [f'CH{i:02d}' for i in range(10)]
                ch_types = ['seeg'] * 10  # sEEG channel type
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                
                # Create Raw object (data should be in volts for MNE, but mock data is fine)
                mock_obj = mne.io.RawArray(mock_data, info)
                
                filename = f"{label_name}_recording_{file_idx:02d}.pkl"
                filepath = os.path.join(patient_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(mock_obj, f)
    
    print("✓ Mock data created successfully")
    return mock_data_dir


def main():
    """Run all tests"""
    print("Starting iEEG DataLoader Tests")
    print("=" * 50)
    
    # Create mock data for testing

    create_mock_data(MOCK_DATA_DIR)


    # Run tests
    tests = [
        # test_resampling_functionality,
        # Uncomment these after updating the raw_data_path in the functions
        test_ieeg_dataset,
        # test_ieeg_datamodule,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
            print("✓ PASSED")
        else:
            print("✗ FAILED")
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(MOCK_DATA_DIR)
        shutil.rmtree(f"{TMP_LOC}/test_mock_processed")
        shutil.rmtree(f"{TMP_LOC}/test_ieeg_processed", ignore_errors=True)
        shutil.rmtree(f"{TMP_LOC}/test_resample", ignore_errors=True)
    except:
        pass


if __name__ == "__main__":
    main()
