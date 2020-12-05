#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include "imcore.h"
#include "cvcore.h"

int main() 
{
    // read the input image
    matrix_t *img = imread("../data/example.bmp");
    matrix_t *gray = matrix_create(uint8_t, rows(img), cols(img), 1);

    // convert the input into grayscale
    rgb2gray(img, gray);

    // create a feature model
    struct feature_t *feature_extractor[4];

    // get the feature extractor
    feature_extractor[0] = feature_create(CV_ENCODER, cols(img), rows(img), 1, "");
    feature_extractor[1] = feature_create(CV_NPD, cols(img), rows(img), 1, "-n_sample:100");
    feature_extractor[2] = feature_create(CV_LBP, cols(img), rows(img), 1, "-block:10x10 -uniform:3");
    feature_extractor[3] = feature_create(CV_HOG, cols(img), rows(img), 1, "-block:2x2 -cell:10x10 -stride:1x1");

    // allocate space for the feature vector
    char filename[256];
    char *feature_name[4] = {"encoder", "npd", "lbp", "hog"};
    int f = 0;
    for(f = 0; f < 4; f++)
    {
        float *feature = (float *)calloc(feature_size(feature_extractor[f]), sizeof(float));
  
        // print the class names and files under them
        feature_extract(gray, feature_extractor[f], feature);

        // display the information about the feature
        feature_view(feature_extractor[f]);

        // visualize the feature
        matrix_t *visual = feature_visualize(feature, feature_extractor[f]);

        // write the result
        sprintf(filename, "%s_result.bmp", feature_name[f]);
        imwrite(visual, filename);

        // free memory
        free(feature);
    }

    return 0;
}
