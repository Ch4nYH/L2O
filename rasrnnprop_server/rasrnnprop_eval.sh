
python -u evaluate.py \
--problem=rastrigin \
--optimizer=L2L \
--path=./rasrnnprop_n2 \
--num_epochs=1 \
--num_steps=1000 \
--bs=1280 \
--n=2 \
--seed=0

python -u evaluate.py \
--problem=rastrigin \
--optimizer=L2L \
--path=./rasrnnprop_n10 \
--num_epochs=1 \
--num_steps=1000 \
--bs=1280 \
--n=10 \
--seed=0