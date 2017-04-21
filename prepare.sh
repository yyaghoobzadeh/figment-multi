DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#first add the project directory to your PYTHONPATH
#This script should be run only once, before starting the experiments, to build the datastreams
cd data
wget http://cistern.cis.lmu.de/figment2/embeddings/skip,200dim.txt
wget http://cistern.cis.lmu.de/figment2/embeddings/sskip,200dim.txt
wget http://cistern.cis.lmu.de/figment2/embeddings/fasttext,200dim.txt
	
echo "building fuel datasets for skip..."

mkdir skip
cd skip
cp ../datasets/* .

mv ../skip,200dim.txt .
python ../../src/emb2numpy.py skip,200dim.txt 
wait
cp ../config .
echo "ent_vectors=skip,200dim.txt.npy" >> config
echo "dsdir="$DIR/data/skip/ >> config
python ../../src/make_dataset.py config &
wait
rm Etrain.names Edev.names Etest.names types
mv skip,200dim.txt ../.

echo "building fuel datasets for sskip and fasttext..."
cd ..
mkdir sskip
cd sskip
cp ../datasets/* .
mv ../sskip,200dim.txt .
mv ../fasttext,200dim.txt .
python ../../src/emb2numpy.py sskip,200dim.txt 
wait
python ../../src/emb2numpy.py fasttext,200dim.txt 
wait

cp ../config .
echo "ent_vectors=sskip,200dim.txt.npy" >> config
echo "fasttext_vecfile=fasttext,200dim.txt.npy" >> config
echo "dsdir="$DIR/data/sskip/ >> config
python ../../src/make_dataset.py config 
wait
rm Etrain.names Edev.names Etest.names types
mv sskip,200dim.txt ../.
mv fasttext,200dim.txt ../.
