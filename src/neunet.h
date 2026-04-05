#ifndef HDRNN_NEUNET_H
#define HDRNN_NEUNET_H

#include <stddef.h>
#include <stdint.h>

struct nn_layer
{
    uint32_t n_in; // Number of inputs
    uint32_t n_out; // Number of outputs
    float *w; // Weight matrix
    float *b; // Bias vector
    float *z; // Weighted sums before activation
    float *a; // Activated outputs
};

extern int nn_init_layer(uint32_t n_in, uint32_t n_out,
                         struct nn_layer *layer);

extern void nn_free_layer(struct nn_layer *layer);

extern float *nn_forward_layer(struct nn_layer *layer, const float *x);

extern float *nn_forward(struct nn_layer *layers, uint8_t n_layers,
                         const float *x);

extern size_t nn_argmax(const float *a, size_t n);

extern void nn_softmax(float *a, size_t n);

extern float nn_loss(const float *softmax, size_t i);

extern int nn_backprop(struct nn_layer *layers, uint8_t n_layers,
                       const float *x, size_t y_idx, float lr);

#endif // HDRNN_NEUNET_H
