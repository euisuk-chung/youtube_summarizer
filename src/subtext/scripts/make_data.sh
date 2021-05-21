# params
max_window=4
embed_type='bert'

# run code in for-loop
for ws in $(eval echo "{1..$max_window}")
do
    python make_data.py \
            --window_size ${ws} \
            --embed_type ${embed_type}
done