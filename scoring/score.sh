rm -r scores

mkdir -p scores/de
mkdir -p scores/es
mkdir -p scores/cs
mkdir -p scores/ru
mkdir -p scores/uk

echo "de bleu"
python -u bleu.py de | grep -v -- "-$" > scores/de/bleu
python -u simple-bleu.py de | grep -v -- "-$" > scores/de/simple-bleu
python -u interpolate-bleu.py de | grep -v -- "-$" > scores/de/interpolate-bleu

echo "es bleu"
python -u bleu.py es | grep -v -- "-$" > scores/es/bleu
python -u simple-bleu.py es | grep -v -- "-$" > scores/es/simple-bleu

echo "cs bleu"
python -u bleu.py cs | grep -v -- "-$" > scores/cs/bleu
python -u simple-bleu.py cs | grep -v -- "-$"  > scores/cs/simple-bleu

echo "ru bleu"
python -u bleu.py ru | grep -v -- "-$" > scores/ru/bleu
python -u simple-bleu.py ru | grep -v -- "-$" > scores/ru/simple-bleu

echo "uk bleu"
python -u bleu.py uk | grep -v -- "-$" > scores/uk/bleu
python -u simple-bleu.py uk | grep -v -- "-$" > scores/uk/simple-bleu


echo "de comet"
python -u comet.py de | grep -v -- "-$" > scores/de/comet
python -u simple-comet.py de | grep -v -- "-$" > scores/de/simple-comet
python -u interpolate-comet.py de | grep -v -- "-$" > scores/de/interpolate-comet

echo "es comet"
python -u comet.py es | grep -v -- "-$" > scores/es/comet
python -u simple-comet.py es | grep -v -- "-$" > scores/es/simple-comet

echo "cs comet"
python -u comet.py cs | grep -v -- "-$" > scores/cs/comet
python -u simple-comet.py cs | grep -v -- "-$"  > scores/cs/simple-comet

echo "ru comet"
python -u comet.py ru | grep -v -- "-$" > scores/ru/comet
python -u simple-comet.py ru | grep -v -- "-$" > scores/ru/simple-comet

echo "uk comet"
python -u comet.py uk | grep -v -- "-$" > scores/uk/comet
python -u simple-comet.py uk | grep -v -- "-$" > scores/uk/simple-comet
