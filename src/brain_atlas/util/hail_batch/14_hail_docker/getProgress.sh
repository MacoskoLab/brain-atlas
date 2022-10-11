cd /dev/shm

while true; do
    if compgen -G "/dev/shm/rtmp*" > /dev/null; then
        echo "Some files exist."
        break
    fi
    sleep 30
done

TOTAL=$(ls|grep rtmp|head -n 1|sed -E "s/(^[^_]*_)|(__.*)//g")
echo $TOTAL
# Assume divide by 10
while true; do ls rtmp*|wc -l; sleep 10; done|tqdm --ncols=100  --update_to --total=$TOTAL --smoothing=0.3 --initial=`ls rtmp* |wc -l`
