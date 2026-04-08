#include "neunet.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Kaiming He's uniform distribution.
 * @see https://en.wikipedia.org/wiki/Weight_initialization#He_initialization
 */
static float he_init(uint32_t n_in) { return sqrtf(6.0f / (float) n_in); }

/**
 * @brief Rectified linear unit
 * @see https://en.wikipedia.org/wiki/Rectified_linear_unit
 */
static float relu(float x) { return x > 0.0f ? x : 0.0f; }

static float max_a(const float *a, size_t n)
{
    float max = a[0];
    for (size_t i = 1; i < n; ++i)
        max = a[i] > max ? a[i] : max;

    return max;
}

int nn_init_layer(uint32_t n_in, uint32_t n_out, struct nn_layer *layer)
{
    // Required to free NULL pointers safely
    memset(layer, 0, sizeof(struct nn_layer));

    layer->w = calloc(n_in * n_out, sizeof(float));
    if (layer->w == NULL)
        goto FAIL;

    for (uint32_t i = 0; i < n_in * n_out; ++i) {
        // Inspired by https://stackoverflow.com/a/13409133
        const float r = (float) rand() / (float) RAND_MAX * 2.0f - 1.0f;
        layer->w[i] = r * he_init(n_in);
    }

    layer->b = calloc(n_out, sizeof(float));
    if (layer->b == NULL)
        goto FAIL;

    layer->z = malloc(sizeof(float) * n_out);
    if (layer->z == NULL)
        goto FAIL;

    layer->a = malloc(sizeof(float) * n_out);
    if (layer->a == NULL)
        goto FAIL;

    layer->n_in = n_in;
    layer->n_out = n_out;
    return 0;

FAIL:
    free(layer->w);
    free(layer->b);
    free(layer->z);
    free(layer->a);
    memset(layer, 0, sizeof(struct nn_layer));
    return -1;
}

void nn_free_layer(struct nn_layer *layer)
{
    free(layer->w);
    free(layer->b);
    free(layer->z);
    free(layer->a);
    memset(layer, 0, sizeof(struct nn_layer));
}

float *nn_forward(struct nn_layer *layers, uint8_t n_layers, const float *x)
{
    for (uint8_t i = 0; i < n_layers; ++i) {
        struct nn_layer *layer = &layers[i];

        for (size_t j = 0; j < layer->n_out; ++j) {
            layer->z[j] = 0.0f;
            for (size_t k = 0; k < layer->n_in; ++k)
                layer->z[j] += layer->w[j * layer->n_in + k] * x[k];

            layer->z[j] += layer->b[j];

            // Apply ReLU only to hidden layers
            layer->a[j] = layer->z[j];
            if (i != n_layers - 1)
                layer->a[j] = relu(layer->a[j]);
        }

        x = layer->a;
    }

    return (float *) x;
}

size_t nn_argmax(const float *a, size_t n)
{
    size_t i = 0;
    float max = a[0];

    for (size_t j = 1; j < n; ++j) {
        if (a[j] > max) {
            max = a[j];
            i = j;
        }
    }

    return i;
}

void nn_softmax(float *a, size_t n)
{
    float sum = 0, m = max_a(a, n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = expf(a[i] - m);
        sum += a[i];
    }

    for (size_t i = 0; i < n; ++i)
        a[i] /= sum;
}

float nn_loss(const float *softmax, size_t i)
{
    return -logf(fmaxf(softmax[i], 1e-7f));
}

/**
 * @param lr learning rate (0...1)
 */
int nn_backprop(struct nn_layer *layers, uint8_t n_layers, const float *x,
                size_t y, float lr)
{
    size_t max = 0; // Max layer size (safe upper bound)
    float *delta = NULL, *delta_prev = NULL;
    struct nn_layer *out = &layers[n_layers - 1];

    for (uint8_t i = 0; i < n_layers; ++i) {
        if (layers[i].n_out > max)
            max = layers[i].n_out;
    }

    delta = malloc(sizeof(float) * max);
    if (delta == NULL)
        goto FAIL;

    delta_prev = malloc(sizeof(float) * max);
    if (delta_prev == NULL)
        goto FAIL;

    // Output delta
    for (size_t i = 0; i < out->n_out; ++i)
        delta[i] = out->a[i];

    delta[y] -= 1.0f;

    // Backward
    for (int l = n_layers - 1; l >= 0; --l) {
        struct nn_layer *layer = &layers[l];
        struct nn_layer *prev = NULL;
        const float *input = l == 0 ? x : layers[l - 1].a;

        // Update weights
        for (size_t i = 0; i < layer->n_out; ++i) {
            for (size_t j = 0; j < layer->n_in; ++j) {
                layer->w[i * layer->n_in + j] -= lr * delta[i] * input[j];
            }
            layer->b[i] -= lr * delta[i];
        }

        if (l == 0)
            break;

        // Compute delta_prev
        prev = &layers[l - 1];
        for (size_t j = 0; j < prev->n_out; ++j) {
            delta_prev[j] = 0.0f;

            for (size_t i = 0; i < layer->n_out; ++i) {
                delta_prev[j] += delta[i] * layer->w[i * layer->n_in + j];
            }

            // ReLU derivative
            if (prev->z[j] <= 0.0f)
                delta_prev[j] = 0.0f;
        }

        // Copy delta_prev into delta
        for (size_t j = 0; j < prev->n_out; ++j)
            delta[j] = delta_prev[j];
    }

    free(delta);
    free(delta_prev);
    return 0;

FAIL:
    free(delta);
    free(delta_prev);
    return -1;
}
