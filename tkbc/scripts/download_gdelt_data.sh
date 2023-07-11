cd ../src_data/
mkdir GDELT
cd GDELT
wget https://raw.githubusercontent.com/manuel-dileo/BoxTE/main/datasets/GDELT/train.txt
wget https://raw.githubusercontent.com/manuel-dileo/BoxTE/main/datasets/GDELT/valid.txt
wget https://raw.githubusercontent.com/manuel-dileo/BoxTE/main/datasets/GDELT/test.txt
sed -s 's/\s\{1,\}/\t/g' train.txt > train
sed -s 's/\s\{1,\}/\t/g' valid.txt > valid
sed -s 's/\s\{1,\}/\t/g' test.txt > test