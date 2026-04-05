#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataset.h"
#include "neunet.h"

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

    srand(time(NULL));
    ds_shuffle(&ds);

    printf("First shuffled 50 labels:\n");
    for (size_t i = 0; i < 50; ++i)
        printf("%u%s", ds.y[i], (i + 1) % 25 == 0 ? "\n" : ", ");

    printf("First normalized image after shuffle:\n");
    for (size_t i = 0; i < ds.n; ++i)
        printf("%.1f%s", ds.x[i], (i + 1) % 14 == 0 ? "\n" : ", ");

    struct nn_layer l[3];
    nn_init_layer(ds.n, 128, &l[0]);
    nn_init_layer(128, 64, &l[1]);
    nn_init_layer(64, 10, &l[2]);

    const float *a = nn_forward(l, 3, ds.x);
    for (size_t i = 0; i < l[2].n_out; ++i)
        printf("%lu ~ %.2f\n", i, a[i]);

    a = nn_softmax(a, l[2].n_out);
    for (size_t i = 0; i < l[2].n_out; ++i)
        printf("softmax: %lu ~ %.2f\n", i, a[i]);
    printf("argmax: %.2f\n", a[nn_argmax(a, l[2].n_out)]);
    free((float *) a);

    nn_free_layer(&l[0]);
    nn_free_layer(&l[1]);
    nn_free_layer(&l[2]);
    ds_free(&ds);
    return 0;
}
