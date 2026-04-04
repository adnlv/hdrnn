#include <stdio.h>

#include "dataset.h"

int main(void)
{
    struct dataset ds;

    if (ds_load_mnist_labels("assets/train-labels-idx1-ubyte", &ds) != 0) {
        perror("Error: ds_load_mnist_labels");
        return 1;
    }

    if (ds_load_mnist_images("assets/train-images-idx3-ubyte", &ds) != 0) {
        perror("Error: ds_load_mnist_images");
        return 1;
    }

    printf("Number of images in the dataset: %u\n", ds.c);
    printf("Length of each image: %u\n", ds.n);

    printf("First 50 labels:\n");
    for (size_t i = 0; i < 50; ++i)
        printf("%u%s", ds.y[i], (i + 1) % 25 == 0 ? "\n" : ", ");

    printf("First normalized image:\n");
    for (size_t i = 0; i < ds.n; ++i)
        printf("%.1f%s", ds.x[i], (i + 1) % 14 == 0 ? "\n" : ", ");

    ds_free(&ds);
    return 0;
}
