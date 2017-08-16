#include "transfer_functions.h"
#include <math.h>		// exp, sqrt, tanh

inline
void transfer_v_sigmoid( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;

	for( i = 0; i < len; ++i ) {
		dst[ i ] = 1.0f / ( 1.0f + expf( -src[ i ] ) );
	}
}

inline
void transfer_v_sigmoid_derivative( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;
	for( i = 0; i < len; ++i ) {

		float s = 1.0f / ( 1.0f + expf( -src[ i ] ) );
		dst[ i ] = s * ( 1.0f - s );
	}
}

inline
void transfer_v_rational_sigmoid( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;

	for( i = 0; i < len; ++i ) {
		float f = src[ i ];
		dst[ i ] = f / ( 1.0f + sqrtf( 1.0f + f * f ) );
	}
}

inline
void transfer_v_rational_sigmoid_derivative( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;

	for( i = 0; i < len; ++i ) {
		float f = src[ i ];
		float s = sqrtf( 1.0f + f * f  );
		dst[ i ] = 1.0f / ( s * ( 1.0f + s ) );
	}
}

inline
void transfer_v_linear( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;

	for( i = 0; i < len; ++i ) {
		dst[ i ] = src[ i ];
	}
}

inline
void transfer_v_linear_derivative( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;

	for( i = 0; i < len; ++i ) {
		dst[ i ] = 1.0f;
	}
}

inline
void transfer_v_gaussian( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;

	for( i = 0; i < len; ++i ) {
		float g = src[ i ];
		dst[ i ] = expf( -( g * g ) );
	}
}

inline
void transfer_v_gaussian_derivative( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;

	for( i = 0; i < len; ++i ) {
		float g = src[ i ];
		float f = expf( -( g * g ) );
		dst[ i ] = -2.0f * g * f;
	}
}

inline
void transfer_v_tanh( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;

	for( i = 0; i < len; ++i ) {
		dst[ i ] = tanhf( src[ i ] );
	}
}

inline
void transfer_v_tanh_derivative( int width, int height, const float *src, float *dst ) {
	int i;
	int len = width * height;

	for( i = 0; i < len; ++i ) {
		float t = tanhf( src[ i ] );
		dst[ i ] = 1.0f - t * t;
	}
}
