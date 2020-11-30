
python -u train_rnnprop_cl_mt.py \
--problem=rastrigin \
--bs=1280 \
--n=2 \
--save_path=./rasrnnprop_n2 \
--num_epochs=10000 \
--mt_ratio=0.3 \
--evaluation_period=10 \
--init="normal"

python -u train_rnnprop_cl_mt.py \
--problem=rastrigin \
--bs=1280 \
--n=10 \
--save_path=./rasrnnprop_n10 \
--num_epochs=10000 \
--mt_ratio=0.3 \
--evaluation_period=10 \
--init="normal"
