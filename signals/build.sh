gcc -I/home/cyrille/GIT/Viz/datoviz/include \
    -I/home/cyrille/GIT/Viz/datoviz/build/_deps/cglm-src/include/ \
    -L/home/cyrille/GIT/Viz/datoviz/build \
    slimshady.c -o slimshady -ldatoviz
