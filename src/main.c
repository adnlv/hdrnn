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

    srand(time(NULL));
    ds_shuffle(&ds);

    // Initialize network
    struct nn_layer l[3];
    nn_init_layer(ds.n, 128, &l[0]);
    nn_init_layer(128, 64, &l[1]);
    nn_init_layer(64, 10, &l[2]);

    size_t label = ds.y[0];
    const float *x = ds.x;

    printf("Correct label: %lu\n", label);

    // Before training
    float *a = nn_forward(l, 3, x);
    nn_softmax(a, l[2].n_out);

    float loss_before = nn_loss(a, label);
    size_t pred_before = nn_argmax(a, l[2].n_out);

    printf("Before training:\n");
    printf("\tloss = %f\n", loss_before);
    printf("\tprediction = %lu\n", pred_before);

    // Train
    const int steps = 200;
    for (int i = 0; i < steps; ++i) {
        a = nn_forward(l, 3, x);
        nn_softmax(a, l[2].n_out);
        nn_backprop(l, 3, x, label, 0.01f);
    }

    // After training
    a = nn_forward(l, 3, x);
    nn_softmax(a, l[2].n_out);

    float loss_after = nn_loss(a, label);
    size_t pred_after = nn_argmax(a, l[2].n_out);

    printf("After training:\n");
    printf("\tloss = %f\n", loss_after);
    printf("\tprediction = %lu\n", pred_after);

    // Cleanup
    nn_free_layer(&l[0]);
    nn_free_layer(&l[1]);
    nn_free_layer(&l[2]);
    ds_free(&ds);
    return 0;
}
