#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataset.h"
#include "neunet.h"

int main(void)
{
    size_t epochs = 3, n_samples = 10000, limit, label;
    float lr = 0.0005f, *x = NULL, *a = NULL;
    struct dataset ds;
    struct nn_layer l[3];

    if (ds_load_mnist_labels("assets/train-labels-idx1-ubyte", &ds) != 0) {
        perror("Error: ds_load_mnist_labels");
        return 1;
    }

    if (ds_load_mnist_images("assets/train-images-idx3-ubyte", &ds) != 0) {
        perror("Error: ds_load_mnist_images");
        return 1;
    }

    limit = ds.c;

    srand(time(NULL));

    // Initialize network
    nn_init_layer(ds.n, 128, &l[0]);
    nn_init_layer(128, 64, &l[1]);
    nn_init_layer(64, 10, &l[2]);

    for (size_t e = 0; e < epochs; ++e) {
        float total_loss = 0.0f;
        size_t correct = 0;

        ds_shuffle(&ds);

        for (size_t i = 0; i < limit; ++i) {
            label = ds.y[i];
            x = ds.x + i * ds.n;
            a = nn_forward(l, 3, x);

            nn_softmax(a, l[2].n_out);

            total_loss += nn_loss(a, label);

            if (nn_argmax(a, l[2].n_out) == label)
                correct++;

            nn_backprop(l, 3, x, label, lr);

            if ((i + n_samples) % n_samples == 0)
                printf("sample %lu loss = %f\n", i, nn_loss(a, label));
        }

        printf("epoch %lu:\n", e);
        printf("\tavg loss = %f\n", total_loss / (float) limit);
        printf("\taccuracy = %.2f%%\n",
               100.0f * (double) correct / (double) limit);
    }

    // Final check on first sample
    x = ds.x;
    label = ds.y[0];
    a = nn_forward(l, 3, x);

    nn_softmax(a, l[2].n_out);

    printf("Final sample check:\n");
    printf("\tloss = %f\n", nn_loss(a, label));
    printf("\tprediction = %lu (label = %lu)\n", nn_argmax(a, l[2].n_out),
           label);

    // Cleanup
    nn_free_layer(&l[0]);
    nn_free_layer(&l[1]);
    nn_free_layer(&l[2]);
    ds_free(&ds);
    return 0;
}
