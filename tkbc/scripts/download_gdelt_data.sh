cd ../src_data/
mkdir GDELT
cd GDELT
wget https://raw.githubusercontent.com/manuel-dileo/BoxTE/main/datasets/GDELT/train.txt
wget https://raw.githubusercontent.com/manuel-dileo/BoxTE/main/datasets/GDELT/valid.txt
wget https://raw.githubusercontent.com/manuel-dileo/BoxTE/main/datasets/GDELT/test.txt
mv train.txt train
mv valid.txt valid
mv test.txt test