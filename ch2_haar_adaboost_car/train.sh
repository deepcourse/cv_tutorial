mkdir -p save_dir
opencv_traincascade -data save_dir -vec positives.vec -bg dataset/negatives.txt -numPos 420 -numNeg 450 -w 24 -h 24 -precalcValBufSize 1024 -precalcIdxBufSize 1024 -numStages 25 -acceptanceRatioBreakValue 1.0e-5
