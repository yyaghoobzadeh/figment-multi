#train, test and evaluate a specific model, given by argument
mymodel=$1

cd configs/mymodel/
python ../../src/train_test.py --config config --train t --test t --eval t
#the results are in 'config.meas.ents' and 'config.meas.types'
