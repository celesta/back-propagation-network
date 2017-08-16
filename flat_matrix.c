#include "macros.h"
#include "flat_matrix.h"
#include <assert.h>

/*
	Flat indexing
*/
inline // 1d index of 2d matrix: for col-major form use height, else width.
int idx2d( int dim, int i, int j ) {
	return i * dim + j;
}

inline // 1d index of 3d matrix.
int idx3d( int height, int depth, int i, int j, int k ) {
	return depth * ( i * height + j ) + k;
}

/*
	2d matrix
*/
inline
float matrix_2d_get( const float *m, int dim, int i, int j ) {
	return m[ idx2d( dim, i, j ) ];
}

inline
void matrix_2d_set( float *m, int dim, int i, int j, float value ) {
	m[ idx2d( dim, i, j ) ] = value;
}

inline
void matrix_2d_transpose( const Matrix *src, Matrix *dst ) {
	int i, j;
	int width = src->shape[ 0 ];
	int height = src->shape[ 1 ];

	assert( ( src->shape[ 0 ] == dst->shape[ 1 ] ) && ( src->shape[ 1 ] == dst->shape[ 0 ] ) );

	for( i = 0; i < width; ++i ) {
		for( j = 0; j < height; ++j ) {
			matrix_2d_set( dst->data, width, j, i,
				matrix_2d_get( src->data, height, i, j ) );
		}
	}
}

inline // [ k, l ] x [ l, m ] -> [ k, m ]
void matrix_2d_mul( const Matrix *a, const Matrix *b, Matrix *dst ) {
	int i, j, k;
	int wa = a->shape[ 0 ];
	int ha = a->shape[ 1 ];
	int hb = b->shape[ 1 ];

	assert( a->shape[ 1 ] == b->shape[ 0 ] );
	assert( ( a->shape[ 0 ] == dst->shape[ 0 ] ) && ( b->shape[ 1 ] == dst->shape[ 1 ] ) );

	for( i = 0; i < wa; ++i ) {
		for( j = 0; j < hb; ++j ) {
			float sum = 0.0f;

			for( k = 0; k < ha; ++k ) {
				sum += matrix_2d_get( a->data, ha, i, k )
					* matrix_2d_get( b->data, hb, k, j );
			}
			matrix_2d_set( dst->data, hb, i, j, sum );
		}
	}
}

inline
void matrix_2d_mul_elements( const Matrix *a, const Matrix *b, Matrix *dst ) {
	int i, j, idx;
	int width = a->shape[ 0 ];
	int height = a->shape[ 1 ];

	assert( ( a->shape[ 0 ] == b->shape[ 0 ] ) && ( a->shape[ 1 ] == b->shape[ 1 ] ) );

	for( i = 0; i < width; i++ ) {
		for( j = 0; j < height; j++ ) {
			idx = idx2d( height, i, j );
			dst->data[ idx ] = a->data[ idx ] * b->data[ idx ];
		}
	}
}

inline
void matrix_2d_mul_scalar( const Matrix *src, Matrix *dst, float scalar ) {
	int i;
	int len = src->shape[ 0 ] * src->shape[ 1 ];

	assert( ( src->shape[ 0 ] == dst->shape[ 0 ] ) && ( src->shape[ 1 ] == dst->shape[ 1 ] ) );

	for( i = 0; i < len; ++i ) {
		dst->data[ i ] = src->data[ i ] * scalar;
	}
}

inline
void matrix_2d_mul_scalar_self( Matrix *m, float scalar ) {
	int i;
	int len = m->shape[ 0 ] * m->shape[ 1 ];

	for( i = 0; i < len; ++i ) {
		m->data[ i ] *= scalar;
	}
}

inline
void matrix_2d_add( const Matrix *a, const Matrix *b, Matrix *dst ) {
	int i;
	int len = a->shape[ 0 ] * a->shape[ 1 ];

	assert( ( a->shape[ 0 ] == b->shape[ 0 ] ) && ( a->shape[ 0 ] == dst->shape[ 0 ] ) );
	assert( ( a->shape[ 1 ] == b->shape[ 1 ] ) && ( a->shape[ 1 ] == dst->shape[ 1 ] ) );

	for( i = 0; i < len; ++i ) {
		dst->data[ i ] = a->data[ i ] + b->data[ i ];
	}
}

inline
void matrix_2d_sub( const Matrix *a, const Matrix *b, Matrix *dst ) {
	int i;
	int len = a->shape[ 0 ] * a->shape[ 1 ];

	assert( ( a->shape[ 0 ] == b->shape[ 0 ] ) && ( a->shape[ 0 ] == dst->shape[ 0 ] ) );
	assert( ( a->shape[ 1 ] == b->shape[ 1 ] ) && ( a->shape[ 1 ] == dst->shape[ 1 ] ) );

	for( i = 0; i < len; ++i ) {
		dst->data[ i ] = a->data[ i ] - b->data[ i ];
	}
}

inline
void matrix_2d_sub_self( Matrix *self, const Matrix *other ) {
	int i;
	int len = self->shape[ 0 ] * self->shape[ 1 ];

	assert( ( self->shape[ 0 ] == other->shape[ 0 ] ) && ( self->shape[ 1 ] == other->shape[ 1 ] ) );

	for( i = 0; i < len; ++i ) {
		self->data[ i ] -= other->data[ i ];
	}
}

inline
void matrix_2d_square( const Matrix *src, Matrix *dst ) {
	int i;
	int len = src->shape[ 0 ] * src->shape[ 1 ];

	assert( ( src->shape[ 0 ] == dst->shape[ 0 ] ) && ( src->shape[ 1 ] == dst->shape[ 1 ] ) );

	for( i = 0; i < len; ++i ) {
		float f = src->data[ i ];
		dst->data[ i ] = f * f;
	}
}

inline
float matrix_2d_sum( const Matrix *m ) {
	float sum = 0.0f;
	int i;
	int len = m->shape[ 0 ] * m->shape[ 1 ];

	for( i = 0; i < len; ++i ) {
		sum += m->data[ i ];
	}
	return sum;
}

inline
void matrix_2d_slice_row( const Matrix *src, Matrix *dst ) {
	int i, j;
	int wa = src->shape[ 0 ];
	int ha = src->shape[ 1 ];
	int hb = dst->shape[ 1 ];

	assert( ( ( wa - 1 ) == dst->shape[ 0 ] ) && ( ha == hb ) );

	for( i = 0; i < wa; ++i ) {
		for( j = 0; j < ha; ++j ) {
			matrix_2d_set( dst->data, hb, i, j,
				matrix_2d_get( src->data, ha, i, j ) );
		}
	}
}

inline
void matrix_2d_append_ones( const Matrix *src, Matrix *dst ) {
	int i;
	int len1 = src->shape[ 0 ] * src->shape[ 1 ];
	int len2 = len1 + src->shape[ 1 ];

	assert( ( ( src->shape[ 0 ] + 1 ) == dst->shape[ 0 ] )
		&& ( src->shape[ 1 ] == dst->shape[ 1 ] ) );

	for( i = 0; i < len2; ++i ) {
		if( i < len1 ) {
			dst->data[ i ] = src->data[ i ];
		} else {
			dst->data[ i ] = 1.0f;
		}
	}
}

/*
	3d matrix
*/
inline
float matrix_3d_get( const float *m, int height, int depth, int i, int j, int k ) {
	return m[ idx3d( height, depth, i, j, k ) ];
}

inline
void matrix_3d_set( float *m, int height, int depth, int i, int j, int k, float value ) {
	m[ idx3d( height, depth, i, j, k ) ] = value;
}

inline
void matrix_3d_from_2d( const Matrix *src, Matrix *dst ) {
	int i, j;
	int width = src->shape[ 0 ];
	int height = src->shape[ 1 ];

	assert( ( src->shape[ 0 ] == dst->shape[ 1 ] ) && ( src->shape[ 1 ] == dst->shape[ 2 ] ) );

	for( i = 0; i < width; ++i ) {
		for( j = 0; j < height; ++j ) {
			matrix_3d_set( dst->data, width, height, 0, i, j,
				matrix_2d_get( src->data, height, i, j ) );
		}
	}
}
// Transpose a shape's dimensions [w, h, d] to position [x, y, z]
inline
void matrix_3d_transposed_shape( int w, int h, int d, int x, int y, int z, int *shape )
{
	if( 0 != x ) {
		if( x == 1 ) {
			shape[ 0 ] = h;
		} else {
			shape[ 0 ] = d;
		}
	} else {
		shape[ 0 ] = w;
	}
	if( 1 != y ) {
		if( y == 0 ) {
			shape[ 1 ] = w;
		} else {
			shape[ 1 ] = d;
		}
	} else {
		shape[ 1 ] = h;
	}
	if( 2 != z ) {
		if( z == 0 ) {
			shape[ 2 ] = w;
		} else {
			shape[ 2 ] = h;
		}
	} else {
		shape[ 2 ] = d;
	}
}

inline
void matrix_3d_transpose( int a1, int a2, int a3, const Matrix *src, Matrix *dst ) {
	int a, b, c, i, j, k, shape[ 3 ];
	int w = src->shape[ 0 ];
	int h = src->shape[ 1 ];
	int d = src->shape[ 2 ];

	matrix_3d_transposed_shape( w, h, d, a1, a2, a3, shape );

	assert( ( a1 != a2 ) && ( a1 != a3 ) && ( a2 != a3 ) );
	assert( ( dst->shape[ 0 ] == shape[ 0 ] )
		&& (  dst->shape[ 1 ] == shape[ 1 ]  )
		&& (  dst->shape[ 2 ] == shape[ 2 ] ) );

	for( i = 0; i < w; ++i ) {
		for( j = 0; j < h; ++j ) {
			for( k = 0; k < d; ++k ) {
				if( 0 == a1 ) {
					a = i;
				} else if( 1 == a1 ) {
					a = j;
				} else {
					a = k;
				}
				if( 1 == a2 ) {
					b = j;
				} else if( 0 == a2 ) {
					b = i;
				} else {
					b = k;
				}
				if( 2 == a3 ) {
					c = k;
				} else if( 0 == a3 ) {
					c = i;
				} else {
					c = j;
				}
				float f = matrix_3d_get( src->data, h, d, i, j, k );
				matrix_3d_set( dst->data, shape[ 1 ], shape[ 2 ], a, b, c, f );
			}
		}
	}
}

inline
void matrix_3d_mul( const Matrix *m1, const Matrix *m2, Matrix *dst )
{
	int i, j, k, l, r, a, b, c, sum = 0;
	float f, g;
	int w1 = m1->shape[ 0 ];
	int h1 = m1->shape[ 1 ];
	int d1 = m1->shape[ 2 ];
	int w2 = m2->shape[ 0 ];
	int h2 = m2->shape[ 1 ];
	int d2 = m2->shape[ 2 ];

	a = max( w1, w2 );
	b = max( h1, h2 );
	c = max( d1, d2 );

	assert( ( a == dst->shape[ 0 ] )
		&& ( b == dst->shape[ 1 ] ) && ( c == dst->shape[ 2 ] ) );

	for( i = 0; i < a; ++i ) {
		for( j = 0; j < h1; ++j ) {
			for( k = 0; k < d2; ++k ) {
				for( l = 0; l < d1; ++l ) {
					for( r = 0; r < h2; ++r ) {
						++sum;
						f = matrix_3d_get( m1->data, h1, d1, i, j, l );
						g = matrix_3d_get( m2->data, h2, d2, i, r, j );
						matrix_3d_set( dst->data, b, c, i, r, l, f * g );
					}
				}
			}
		}
	}
	assert( sum == a * b * c );
}

inline
void matrix_3d_sum_along( int axis, const Matrix *src, Matrix *dst ) {
	int i, j, k;
	float sum;
	int w = src->shape[ 0 ];
	int h = src->shape[ 1 ];
	int d = src->shape[ 2 ];

	if( 0 == axis ) {
		assert( ( src->shape[ 1 ] == dst->shape[ 0 ] ) && ( src->shape[ 2 ] == dst->shape[ 1 ] ) );

		for( i = 0; i < h; ++i ) {
			for( j = 0; j < d; ++j ) {
				sum = 0.0f;

				for( k = 0; k < w; ++k ) {
					sum += matrix_3d_get( src->data, h, d, k, i, j );
				}
				matrix_2d_set( dst->data, d, i, j, sum );
			}
		}
	} else if( 1 == axis ) {
		for( i = 0; i < w; ++i ) {
			for( j = 0; j < d; ++j ) {
				sum = 0.0f;

				for( k = 0; k < h; ++k ) {
					sum += matrix_3d_get( src->data, h, d, i, k, j );
				}
				matrix_2d_set( dst->data, d, i, j, sum );
			}
		}
	} else {
		for( i = 0; i < w; ++i ) {
			for( j = 0; j < h; ++j ) {
				sum = 0.0f;

				for( k = 0; k < d; ++k ) {
					sum += matrix_3d_get( src->data, h, d, i, j, k );
				}
				matrix_2d_set( dst->data, h, i, j, sum );
			}
		}
	}
}
