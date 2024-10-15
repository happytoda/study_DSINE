data_path: projects/__init__.py DATASET_DIR

train: projects/dsine/study_train.py

test: projects/dsine/stest_minimal.py

python projects/dsine/study_train.py projects/dsine/experiments/exp111_test/test.txt 
