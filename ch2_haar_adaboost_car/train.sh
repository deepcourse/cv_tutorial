opencv_traincascade -data xml -vec positive.vec -bg neg_small.txt -nstages 5 -nsplits 2 -minhitrate 0.999 -maxfalsealarm 0.5 -h 64 -w 64
