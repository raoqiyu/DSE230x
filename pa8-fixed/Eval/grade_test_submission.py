#!/user/bin/env python3 -tt
"""
If you've reached here and are a student, you should go back. Really.
If caught, it will be treated as an academic integrity violation and may result in suspension. 

Remember, everything is being logged. We have all of your submissions, and their outputs.
"""

import sys
import numpy as np
import pickle
import os

ground_truth_test_labels_small_pkl_path =  "../Data/ground_truth_test_labels_small.pkl"
ground_truth_test_labels_large_pkl_path =  "../Data/ground_truth_test_labels_large.pkl"
test_predictions_student = ""

def main():
    args = sys.argv[1:]
    if(len(args)==2):
        test_predictions_student = args[0]
        dataset_size = args[1]
        
        if(dataset_size == "small"):
            with open(ground_truth_test_labels_small_pkl_path, 'rb') as f:
                gt_labels = np.array(pickle.load(f))
        else:
            with open(ground_truth_test_labels_large_pkl_path, 'rb') as f:
                gt_labels = np.array(pickle.load(f))
        
        with open(test_predictions_student, 'rb') as f:
            test_labels = np.array(pickle.load(f))
        
        print(sum(gt_labels==test_labels)/len(gt_labels))
        sys.exit(0)
    
    else:
        print('usage: grade_test_submission.py test_predictions.pkl small')
        sys.exit(1)

# Main body
if __name__ == '__main__':
    main()