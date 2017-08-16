#ifndef H_FLAT_MATRIX
#define H_FLAT_MATRIX

#include "network.h"

// 2d
float matrix_2d_get( const float *m, int dim, int i, int j );
void matrix_2d_set( float *m, int dim, int i, int j, float value );
void matrix_2d_transpose( const Matrix *src, Matrix *dst );
void matrix_2d_mul( const Matrix *a, const Matrix *b, Matrix *dst );
void matrix_2d_mul_elements( const Matrix *a, const Matrix *b, Matrix *dst );
void matrix_2d_mul_scalar( const Matrix *src, Matrix *dst, float scalar );
void matrix_2d_mul_scalar_self( Matrix *m, float scalar );
void matrix_2d_add( const Matrix *a, const Matrix *b, Matrix *dst );
void matrix_2d_sub( const Matrix *a, const Matrix *b, Matrix *dst );
void matrix_2d_sub_self( Matrix *self, const Matrix *other );
void matrix_2d_square( const Matrix *src, Matrix *dst );
float matrix_2d_sum( const Matrix *m );
void matrix_2d_slice_row( const Matrix *src, Matrix *dst );
void matrix_2d_append_ones( const Matrix *src, Matrix *dst );

// 3d
float matrix_3d_get( const float *m, int height, int depth, int i, int j, int k );
void matrix_3d_set( float *m, int height, int depth, int i, int j, int k, float value );
void matrix_3d_from_2d( const Matrix *src, Matrix *dst );
void matrix_3d_transposed_shape( int w, int h, int d, int x, int y, int z, int *shape );
void matrix_3d_transpose( int a1, int a2, int a3, const Matrix *src, Matrix *dst );
void matrix_3d_mul( const Matrix *m1, const Matrix *m2, Matrix *dst );
void matrix_3d_sum_along( int axis, const Matrix *src, Matrix *dst );

#endif /* H_FLAT_MATRIX */
