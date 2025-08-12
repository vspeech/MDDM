mdl_lst=$1
log_dir_base=$2
test_lst=test.lst.bak
dir=separated/htdemucs

for mdl_file in $(cat $mdl_lst); do
    mdl_fn=$(basename $mdl_file | sed 's/nb-h7654-chsoq8sx-0_2//g' | sed 's/\.pt//g')
    log_dir=${log_dir_base}/${mdl_fn}
    mkdir -p $log_dir
    log_file=$log_dir/log
    score_file=$log_dir/score

    python xspeech/bin/inference.py --train_conf conf/train_unet.yaml --mdl_ckpt ${mdl_file} --test_lst ${test_lst} --shifts 0 --split True --overlap 0.25 --sr 48000 --out_score ${score_file} --out_dir ${log_dir} &> ${log_file}
done
