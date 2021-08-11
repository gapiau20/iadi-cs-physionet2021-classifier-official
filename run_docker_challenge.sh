nvidia-docker run -it \
       -v /mnt/data2/PAUB/CinC2021/model:/physionet/a_generic_directry_containing_the_model/\
       -v /mnt/data2/PAUB/CinC2021/model:/physionet/model\
       -v /mnt/data2/PAUB/CinC2021/tests:/physionet/test_outputs \
       -v /mnt/data1/CinC2021/:/physionet/test_data \
       -v /mnt/data1/CinC2021:/physionet/training_data \
       -p 7779:8888\
       -p 6008:6006\
       -m 50g\
       --cpuset-cpus=0-9\
       --name paub-test_submit_challenge paub-test_challenge_submit:2.0

