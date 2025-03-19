DATOVIZ_FOLDER="../../datoviz"
gcc -I$DATOVIZ_FOLDER/include \
    -I$DATOVIZ_FOLDER/build/_deps/cglm-src/include/ \
    -L$DATOVIZ_FOLDER/build \
    slimshady.c -o slimshady -ldatoviz
