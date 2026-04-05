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

float *nn_forward_layer(struct nn_layer *layer, const float *x)
{
    for (size_t i = 0; i < layer->n_out; ++i) {
        layer->z[i] = 0.0f;
        for (size_t j = 0; j < layer->n_in; ++j)
            layer->z[i] += layer->w[i * layer->n_in + j] * x[j];

        layer->z[i] += layer->b[i];
        layer->a[i] = relu(layer->z[i]);
    }

    return layer->a;
}

float *nn_forward(struct nn_layer *layers, uint8_t n_layers, const float *x)
{
    for (uint8_t i = 0; i < n_layers; ++i)
        x = nn_forward_layer(&layers[i], x);

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

float *nn_softmax(const float *a, size_t n)
{
    float sum = 0, m = max_a(a, n);
    float *out = NULL;

    for (size_t i = 0; i < n; ++i)
        sum += expf(a[i] - m);

    out = malloc(sizeof(float) * n);
    if (out == NULL)
        return NULL;

    for (size_t i = 0; i < n; ++i)
        out[i] = expf(a[i] - m) / sum;

    return out;
}
