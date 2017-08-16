#ifndef H_TRANSFER_FUNCTIONS
#define H_TRANSFER_FUNCTIONS

enum { // Transfer function types
	TF_Gaussian,
	TF_Linear,
	TF_Rational_Sigmoid,
	TF_Sigmoid,
	TF_Tanh
};
// these hopefully will be really vectorized one day

void transfer_v_sigmoid( int width, int height, const float *src, float *dst );

void transfer_v_sigmoid_derivative( int width, int height, const float *src, float *dst );

void transfer_v_rational_sigmoid( int width, int height, const float *src, float *dst );

void transfer_v_rational_sigmoid_derivative( int width, int height, const float *src, float *dst );

void transfer_v_linear( int width, int height, const float *src, float *dst );

void transfer_v_linear_derivative( int width, int height, const float *src, float *dst );

void transfer_v_gaussian( int width, int height, const float *src, float *dst );

void transfer_v_gaussian_derivative( int width, int height, const float *src, float *dst );

void transfer_v_tanh( int width, int height, const float *src, float *dst );

void transfer_v_tanh_derivative( int width, int height, const float *src, float *dst );

#endif /* H_TRANSFER_FUNCTIONS */
