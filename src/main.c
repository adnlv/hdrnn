#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataset.h"
#include "neunet.h"

int main(void)
{
    const char *model_filename = "model.mdl";
    uint8_t num_layers = 3;
    size_t num_epochs = 3;
    size_t num_samples = 10000;
    size_t limit = 0, label = 0;
    float learning_rate = 0.0005f;
    float *pixels_normalized = NULL;
    float *activated_outputs = NULL;
    struct nn_layer *layers = NULL;
    struct dataset dataset;

    if (ds_load_mnist_labels("assets/train-labels-idx1-ubyte", &dataset) != 0) {
        perror("Error: ds_load_mnist_labels");
        return 1;
    }

    if (ds_load_mnist_images("assets/train-images-idx3-ubyte", &dataset) != 0) {
        perror("Error: ds_load_mnist_images");
        return 1;
    }

    limit = dataset.num_images;

    srand((unsigned int)time(NULL));

    // Try to load model
    layers = nn_load(model_filename, &num_layers);
    if (layers == NULL) {
        // Initialize network
        layers = malloc(sizeof(struct nn_layer) * num_layers);
        assert(layers != NULL);

        assert(nn_init_layer(dataset.num_pixels, 128, &layers[0]) == 0);
        assert(nn_init_layer(128, 64, &layers[1]) == 0);
        assert(nn_init_layer(64, 10, &layers[2]) == 0);

        for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
            float total_loss = 0.0f;
            size_t correct = 0;

            ds_shuffle(&dataset);

            for (size_t sample = 0; sample < limit; ++sample) {
                label = dataset.labels[sample];
                pixels_normalized = dataset.pixel_data + sample * dataset.num_pixels;
                activated_outputs = nn_forward(layers, 3, pixels_normalized);

                nn_softmax(activated_outputs, layers[2].n_out);

                total_loss += nn_loss(activated_outputs, label);

                if (nn_argmax(activated_outputs, layers[2].n_out) == label)
                    correct++;

                nn_backprop(layers, 3, pixels_normalized, label, learning_rate);

                if (sample % num_samples == 0)
                    printf("sample %" PRIuMAX " loss = %f\n", num_samples + sample,
                           nn_loss(activated_outputs, label));
            }

            printf("epoch %" PRIuMAX ":\n", epoch);
            printf("\tavg loss = %f\n", total_loss / (float) limit);
            printf("\taccuracy = %.2f%%\n",
                   100.0f * (double) correct / (double) limit);
        }
    }

    // Final check on a single sample - index controls both pixels and label
    const size_t sample_idx = 0;
    pixels_normalized = dataset.pixel_data + sample_idx * dataset.num_pixels;
    label = dataset.labels[sample_idx];
    activated_outputs = nn_forward(layers, 3, pixels_normalized);

    nn_softmax(activated_outputs, layers[2].n_out);

    printf("Final sample check (sample %" PRIuMAX "):\n", sample_idx);
    printf("\tloss = %f\n", nn_loss(activated_outputs, label));
    printf("\tprediction = %" PRIuMAX " (label = %" PRIuMAX ")\n",
           nn_argmax(activated_outputs, layers[2].n_out), label);

    nn_save(model_filename, layers, num_layers);

    // Cleanup
    for (size_t i = 0; i < num_layers; ++i)
        nn_free_layer(&layers[i]);
    
    free(layers);
    ds_free(&dataset);
    return 0;
}
