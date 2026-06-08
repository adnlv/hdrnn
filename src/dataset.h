#ifndef HDRNN_DATASET_H
#define HDRNN_DATASET_H

#include <stdint.h>

struct dataset
{
    uint32_t num_images; // Total number of images in the dataset
    uint32_t num_pixels; // Number of pixels in each image
    float *pixel_data; // Array of normalized pixel data
    uint8_t *labels; // Array of corresponding labels
};

extern int ds_load_mnist_labels(const char *filename, struct dataset *ds);

extern int ds_load_mnist_images(const char *filename, struct dataset *ds);

extern void ds_free(struct dataset *ds);

extern int ds_shuffle(struct dataset *ds);

#endif // HDRNN_DATASET_H
