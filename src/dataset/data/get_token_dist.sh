for i in {0..30};do
    echo $i
    echo $1
    python get_token_dist.py $i $1 &
done