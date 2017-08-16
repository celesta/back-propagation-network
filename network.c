/*
	Back propagation network - gradient descent + momentum

	Note:
		I have no clue about which standard deviation best to chose
		for which transfer functions when generating random numbers
		for initial weights. For Gauss|Tanh in combination with
		Linear 0.1 seems fine.

	TODO

		Vectorized transfer functions
*/
#include "macros.h"
#include "network.h"
#include "transfer_functions.c"
#include "flat_matrix.c"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/user.h>

#define NETWORK_HEADER_LEN		(sizeof( Network ) - sizeof( char * ))

inline
void random_gaussian( float mean, float std_deviation, float *out1,	float *out2 )
{
	float s, t, u, v;

	do {
		u = 2.0f * ( ( float ) rand( ) / ( float ) RAND_MAX ) - 1.0f;
		v = 2.0f * ( ( float ) rand( ) / ( float ) RAND_MAX ) - 1.0f;
		s = u * u + v * v;
	} while( ( s > 1.0f ) || ( ( 0.0f == u ) && ( 0.0f == v ) ) );

	t = sqrtf( ( -2.0f * log( s ) ) / s );
	*out1 = std_deviation * u * t + mean;
	*out2 = std_deviation * v * t + mean;
}

/*
	Matrix shapes
*/
inline
void bpn_weight_shape( int i, int *layer_sizes, int *shape ) {
	shape[ 0 ] = layer_sizes[ i + 1 ];
	shape[ 1 ] = layer_sizes[ i ] + 1; // bias node
}

inline
void bpn_info_get( int dim, int *src, int i, int j, Info *nfo ) {
	int idx = ( idx2d( dim, i, j ) << 2 ); // make it wider to fit in whole Info struct
	nfo->offset		= src[ idx ];
	nfo->shape[ 0 ] = src[ idx + 1 ];
	nfo->shape[ 1 ] = src[ idx + 2 ];
	nfo->shape[ 2 ] = src[ idx + 3 ];
}

inline
void bpn_info_set( int dim, int *dst, int i, int j, Info *nfo ) {
	int idx = ( idx2d( dim, i, j ) << 2 ); // make it wider to fit in whole Info struct
	dst[ idx ]		= nfo->offset;
	dst[ idx + 1 ]	= nfo->shape[ 0 ];
	dst[ idx + 2 ]	= nfo->shape[ 1 ];
	dst[ idx + 3 ]	= nfo->shape[ 2 ];
}

/*
	Prepare matrices for work
*/
inline
void bpn_layer_matrix( View *view, int usage, int layer, int num_layers ) {
	Info nfo;
	bpn_info_get( num_layers, view->info, usage, layer, &nfo );
	Matrix *m = &view->work[ usage ];
	m->data = view->data + nfo.offset;
	m->shape[ 0 ] = nfo.shape[ 0 ];
	m->shape[ 1 ] = nfo.shape[ 1 ];
	m->shape[ 2 ] = nfo.shape[ 2 ];
}

inline
void bpn_special_matrix( View *view, int usage, int layer, int num_layers, int type ) {
	Info nfo;
	bpn_info_get( num_layers, view->info, usage, layer, &nfo );
	Matrix *m = &view->work[ type ];
	m->data = view->data + nfo.offset;
	m->shape[ 0 ] = nfo.shape[ 0 ];
	m->shape[ 1 ] = nfo.shape[ 1 ];
	m->shape[ 2 ] = nfo.shape[ 2 ];
}

inline // Set up a matrix from data and shape
void bpn_general_matrix( float *data, const int *shape, Matrix *out ) {
	out->data = data;
	out->shape[ 0 ] = shape[ 0 ];
	out->shape[ 1 ] = shape[ 1 ];
	out->shape[ 2 ] = shape[ 2 ];
}

/*
	Size requirements
*/
inline
int bpn_get_page_multiple( int bytes ) {
	return ( bytes <= PAGE_SIZE )
		? PAGE_SIZE
		: ( bytes / PAGE_SIZE + 1 ) * PAGE_SIZE;
}

void bpn_get_matrix_sizes( int test_cases, int num_layers, int *sizes,
	size_t *sz_w, size_t *sz_2d, size_t *sz_3d )
{
	int l;
	int n = num_layers - 1;
	int shape[ 2 ];
	size_t a = 0;
	size_t b = 0;
	size_t w = 0;

	for( l = 0; l < num_layers; ++l ) {
		int cur_lsz_bias = sizes[ l ] + 1;
		int next_lsz = sizes[ l + 1 ];
		int tmp = next_lsz * cur_lsz_bias;
		w += tmp;
		a += tmp << 2;						// weight, prev. weight delta, momentum, sum
		a += 5 * next_lsz * test_cases;		// input, output, little delta, derivative, slice
		a += cur_lsz_bias * test_cases;		// stack

		if( n == l ) {
			shape[ 0 ] = test_cases;
			shape[ 1 ] = sizes[ num_layers ];
			// no hidden delta for this layer!
		} else {
			bpn_weight_shape( l + 1, sizes, shape );
			shape[ 0 ] = test_cases;
			a += shape[ 0 ] * shape[ 1 ]; // hidden delta
		}
		a += shape[ 0 ] * shape[ 1 ]; // hidden transpose

		b += 2 * cur_lsz_bias * test_cases;			// output ext., output T
		b += 2 * next_lsz * test_cases;				// little delta ext., little delta T
		b += test_cases * next_lsz * cur_lsz_bias;	// product
	}
	*sz_w = w;
	*sz_2d = a;
	*sz_3d = b;
}

void bpn_get_dimensions( int usage, int layer, int test_cases, int num_layers, int *sizes, int *shape ) {
	int n = num_layers - 1;
	int cur_lsz_bias = sizes[ layer ] + 1;
	int next_lsz = sizes[ layer + 1 ];

	switch( usage ) {
		case Weight:
		case Previous_Weight_Delta:
		case Momentum:
		case Sum: {
			shape[ 0 ] = next_lsz;
			shape[ 1 ] = cur_lsz_bias;
			shape[ 2 ] = 0; // * for 2d set depth to  0
		} break;
		case Layer_Input:
		case Layer_Output:
		case Derivative:
		case Slice:
		case Little_Delta: {
			shape[ 0 ] = next_lsz;
			shape[ 1 ] = test_cases;
			shape[ 2 ] = 0; // *
		} break;
		case Layer_Stack: {
			shape[ 0 ] = cur_lsz_bias;
			shape[ 1 ] = test_cases;
			shape[ 2 ] = 0; // *
		} break;
		case Hidden_Delta:
			if( n == layer ) { // no hidden delta for output layer
				shape[ 0 ] = 0;
				shape[ 1 ] = 0;
				shape[ 2 ] = 0;
				break;
			} // else go on
		case Hidden_T: {
			int tmp_shape[ 2 ];

			if( n == layer ) {
				tmp_shape[ 0 ] = test_cases;
				tmp_shape[ 1 ] = sizes[ num_layers ];
			} else {
				bpn_weight_shape( layer + 1, sizes, tmp_shape );

				if( Hidden_Delta == usage ) {
					tmp_shape[ 0 ] = test_cases;
				}
			}
			shape[ 0 ] = tmp_shape[ 1 ];
			shape[ 1 ] = tmp_shape[ 0 ];
			shape[ 2 ] = 0; // *
		} break;
		case Output_Extended: {
			shape[ 0 ] = 1;
			shape[ 1 ] = cur_lsz_bias;
			shape[ 2 ] = test_cases;
		} break;
		case Output_T: {
			shape[ 0 ] = test_cases;
			shape[ 1 ] = 1;
			shape[ 2 ] = cur_lsz_bias;
		} break;
		case Little_Delta_Extended: {
			shape[ 0 ] = 1;
			shape[ 1 ] = next_lsz;
			shape[ 2 ] = test_cases;
		} break;
		case Little_Delta_T: {
			shape[ 0 ] = test_cases;
			shape[ 1 ] = next_lsz;
			shape[ 2 ] = 1;
		} break;
		case Product: {
			shape[ 0 ] = test_cases;
			shape[ 1 ] = next_lsz;
			shape[ 2 ] = cur_lsz_bias;
		} break;
	}
}

/*
	Network
*/
inline
float bpn_chose_deviation( int transfer_function ) {
	float std_deviation;

	switch( transfer_function ) {
		case TF_Sigmoid: {
			std_deviation = 1.1f;
		} break;
		case TF_Rational_Sigmoid: {
			std_deviation = 1.0f;
		} break;
		case TF_Gaussian: {
			std_deviation = 0.1f;
		} break;
		case TF_Tanh: {
			std_deviation = 0.1f;
		} break;
		case TF_Linear:
		default: {
			std_deviation = 0.1f;
		} break;
	}
	return std_deviation;
};

void bpn_set_weights( View *view, int num_layers, float *values ) {
	int i, j, l;
	Info nfo;
	float *v = values;

	for( l = 0; l < num_layers; ++l ) {
		bpn_info_get( num_layers, view->info, 0, l, &nfo );

		for( i = 0; i < nfo.shape[ 0 ]; ++i ) {
			for( j = 0; j < nfo.shape[ 1 ]; ++j ) {
				matrix_2d_set( view->data + nfo.offset, nfo.shape[ 1 ], i, j, *v++ );
			}
		}
	}
}

void bpn_set_random_weights( View *view, int num_layers ) {
	float deviation, g1, g2;
	int i, j, l;
	Info nfo;

	for( l = 0; l < num_layers; ++l ) {
		bpn_info_get( num_layers, view->info, 0, l, &nfo );

		for( i = 0; i < nfo.shape[ 0 ]; ++i ) {
			// No clue, if this makes any sense
			deviation = bpn_chose_deviation( view->transfer_functions[ i ] );

			for( j = 0; j < nfo.shape[ 1 ]; ++j ) {
				random_gaussian( 0.0f, deviation, &g1, &g2 );
				matrix_2d_set( view->data + nfo.offset, nfo.shape[ 1 ], i, j, g1 );
			}
		}
	}
}

int bpn_allocate( Network *network, View *view, int *layer_sizes, int *transfers ) {
	int i, l;
	int num_layers = network->num_layers;
	int test_cases = network->test_cases;
	int layers2 = num_layers + num_layers; // layer sizes + transfers functions
	int align = layers2 % TEST_BLOCK;
	size_t offset, sz_weight, sz_data, sz_data_2d, sz_data_3d, sz_view, wanted;
	size_t sz_begin = layers2 + align;
	size_t sz_info = ( Usage_Max * num_layers ) << 2; // so it can hold 4 ints
	size_t sz_input_T = test_cases * layer_sizes[ 0 ];
	size_t sz_persist =
		sz_begin * sizeof( int )
		+ sz_info * sizeof( int );

	bpn_get_matrix_sizes( test_cases, num_layers, layer_sizes, &sz_weight, &sz_data_2d, &sz_data_3d );
	sz_data = sz_data_2d + sz_data_3d;
	sz_view = sz_persist
		+ sz_data * sizeof( float )
		+ test_cases * sizeof( float ) // error squared
		+ sz_input_T * sizeof(float );
	wanted = bpn_get_page_multiple( sz_view  );

	char *storage = calloc( wanted, sizeof( char ) );

	if( !storage ) {
		return BPN_No_Memory;
	}

	network->view_bytes = sz_persist;
	network->weight_bytes = sz_weight * sizeof( float ); // want to save weights!
	network->data_bytes = sz_data * sizeof( float );
	network->total_bytes = wanted;
	network->storage = storage;

	view->layer_sizes = ( int * ) storage; // now set pointers in struct view
	offset = num_layers + align / 2;
	view->transfer_functions = ( int * ) storage + offset;
	offset += num_layers + align / 2;
	view->info = ( int * ) storage + offset; // should be 16 byte aligned now
	offset += sz_info;
	view->data = ( float * ) storage + offset;
	offset += sz_data;
	view->error_squared = ( float * ) storage + offset;
	offset += test_cases;
	view->input_T = ( float * ) storage + offset;
	offset += sz_input_T;
	assert( sizeof( int ) * offset == sz_view ); // suppose sizeof int = float

	for( l = 0; l < num_layers; ++l ) { // initialize layer sizes and transfer functions
		view->layer_sizes[ l ] = layer_sizes[ l + 1 ];
		view->transfer_functions[ l ] = transfers[ l ];
	}
	int nfo_offset = 0;

	for( l = 0; l < num_layers; ++l ) { // compute offsets and initialize info structs
		for( i = 0; i < Usage_Max; ++i ) {
			Info nfo;
			bpn_get_dimensions( i, l, test_cases, num_layers, layer_sizes, nfo.shape );
			nfo.offset = nfo_offset;

			if( i < Output_Extended ) { // 2d
				nfo_offset += nfo.shape[ 0 ] * nfo.shape[ 1 ];
			} else { // 3d
				nfo_offset += nfo.shape[ 0 ] * nfo.shape[ 1 ] * nfo.shape[ 2 ];
			}
			bpn_info_set( num_layers, view->info, i, l, &nfo );
		}
	}
	bpn_set_random_weights( view, num_layers );
	return 0;
}

inline
void bpn_apply_transfers( int transfer_function, const Matrix *src, Matrix *dst )
{
	int width = src->shape[ 0 ];
	int height = src->shape[ 1 ];

	switch( transfer_function ) {
		case TF_Sigmoid: {
			transfer_v_sigmoid( width, height, src->data, dst->data );
		} break;
		case TF_Rational_Sigmoid: {
			transfer_v_rational_sigmoid( width, height, src->data, dst->data );
		} break;
		case TF_Gaussian: {
			transfer_v_gaussian( width, height, src->data, dst->data );
		} break;
		case TF_Tanh: {
			transfer_v_tanh( width, height, src->data, dst->data );
		} break;
		case TF_Linear:
		default: {
			transfer_v_linear( width, height, src->data, dst->data );
		} break;
	}
}

inline
void bpn_apply_transfer_derivatives( int transfer_function, const Matrix *src, Matrix *dst )
{
	int width = src->shape[ 0 ];
	int height = src->shape[ 1 ];

	switch( transfer_function ) {
		case TF_Sigmoid: {
			transfer_v_sigmoid_derivative( width, height, src->data, dst->data );
		} break;
		case TF_Rational_Sigmoid: {
			transfer_v_rational_sigmoid_derivative( width, height, src->data, dst->data );
		} break;
		case TF_Gaussian: {
			transfer_v_gaussian_derivative( width, height, src->data, dst->data );
		} break;
		case TF_Tanh: {
			transfer_v_tanh_derivative( width, height, src->data, dst->data );
		} break;
		case TF_Linear:
		default: {
			transfer_v_linear_derivative( width, height, src->data, dst->data );
		} break;
	}
}

void bpn_prepare_run( View *view, int num_inputs, int num_layers, int test_cases, int layer ) {
	bpn_layer_matrix( view, Weight, layer, num_layers );
	bpn_layer_matrix( view, Layer_Input, layer, num_layers );
	bpn_layer_matrix( view, Layer_Output, layer, num_layers );
	bpn_layer_matrix( view, Layer_Stack, layer, num_layers );

	if( layer > 0 ) { // at layer zero get real input, else get output from previous layer
		bpn_special_matrix( view, Layer_Output, layer - 1, num_layers, Input_T );
	} else {
		int shape[ ] = { num_inputs, test_cases, 0 };
		bpn_general_matrix( view->input_T, shape, &view->work[ Input_T ] );
	}
}

void bpn_run( Network *network, View *view, const Matrix *input, Matrix *output ) {
	int l;
	int num_layers = network->num_layers;
	int num_inputs = network->num_inputs;
	int test_cases = network->test_cases;
	Matrix *work = view->work;

	for( l = 0; l < num_layers; ++l ) {
		bpn_prepare_run( view, num_inputs, num_layers, test_cases, l );

		if( 0 == l ) { // need to transpose real input
			matrix_2d_transpose( input, &work[ Input_T ] );
		}
		matrix_2d_append_ones( &work[ Input_T ], &work[ Layer_Stack ] );
		matrix_2d_mul( &work[ Weight ], &work[ Layer_Stack ], &work[ Layer_Input ] );
		bpn_apply_transfers( view->transfer_functions[ l ],
			&work[ Layer_Input ], &work[ Layer_Output ] );
	}
	matrix_2d_transpose( &work[ Layer_Output ], output );
}

void bpn_run_internal( View *view, int num_layers, int num_inputs, int test_cases, const Matrix *input ) {
	int l;
	Matrix *work = view->work;

	for( l = 0; l < num_layers; ++l ) {
		bpn_prepare_run( view, num_inputs, num_layers, test_cases, l );

		if( 0 == l ) { // need to transpose real input
			matrix_2d_transpose( input, &work[ Input_T ] );
		}
		matrix_2d_append_ones( &work[ Input_T ], &work[ Layer_Stack ] );
		matrix_2d_mul( &work[ Weight ], &work[ Layer_Stack ], &work[ Layer_Input ] );
		bpn_apply_transfers( view->transfer_functions[ l ],
			&work[ Layer_Input ], &work[ Layer_Output ] );
	}
}

void bpn_prepare_back( View *view, int num_inputs, int num_layers, int test_cases, int layer, int n ) {
	bpn_layer_matrix( view, Derivative, layer, num_layers );
	bpn_layer_matrix( view, Slice, layer, num_layers );
	bpn_layer_matrix( view, Hidden_T, layer, num_layers );
	bpn_layer_matrix( view, Layer_Input, layer, num_layers );

	if( n == layer ) {
		int shape[ ] = { 1, test_cases, 0 };
		bpn_general_matrix( view->error_squared, shape, &view->work[ Error_Squared ] );
		bpn_layer_matrix( view, Layer_Output, layer, num_layers );
	} else {
		bpn_layer_matrix( view, Little_Delta, layer + 1, num_layers );
		bpn_layer_matrix( view, Weight, layer + 1, num_layers );
		bpn_layer_matrix( view, Hidden_Delta, layer, num_layers );
	}
}

void bpn_prepare_prop( View *view, int num_inputs, int num_layers, int test_cases, int layer, int n ) {
	bpn_layer_matrix( view, Weight, layer, num_layers );
	bpn_layer_matrix( view, Previous_Weight_Delta, layer, num_layers );
	bpn_layer_matrix( view, Momentum, layer, num_layers );
	bpn_layer_matrix( view, Sum, layer, num_layers );
	bpn_layer_matrix( view, Product, layer, num_layers );
	bpn_layer_matrix( view, Output_Extended, layer, num_layers );
	bpn_layer_matrix( view, Output_T, layer, num_layers );
	bpn_layer_matrix( view, Little_Delta, layer, num_layers );
	bpn_layer_matrix( view, Little_Delta_Extended, layer, num_layers );
	bpn_layer_matrix( view, Little_Delta_T, layer, num_layers );
	bpn_layer_matrix( view, Layer_Stack, layer, num_layers );

	if( 0 == layer ) {
		int shape[ ] = { num_inputs, test_cases, 0 };
		bpn_general_matrix( view->input_T, shape, &view->work[ Input_T ] );
	} else {
		bpn_layer_matrix( view, Layer_Output, layer - 1 , num_layers );
	}
}

float bpn_train( Network *network, View *view, const Matrix *input, const Matrix *target ) {
	int l;
	int num_layers = network->num_layers;
	int num_inputs = network->num_inputs;
	int test_cases = network->test_cases;
	const int n = num_layers - 1;
	float error = 1.0f;
	Matrix *work = view->work;

	bpn_run_internal( view, num_layers, num_inputs, test_cases, input );

	// Compute little deltas; apply derivative of transfer function from
	// current layer to matrix of weight deltas downstream, that is,
	// all rows (except the last one (bias)) and all from the columns.
	for( l = n; l >= 0; l-- ) {
		bpn_prepare_back( view, num_inputs, num_layers, test_cases, l, n );

		if( n == l ) { // compare to target values
			matrix_2d_transpose( target, &work[ Hidden_T ] );
			matrix_2d_sub( &work[ Layer_Output ], &work[ Hidden_T ], &work[ Slice ] );
			matrix_2d_square( &work[ Slice ], &work[ Error_Squared ] );
			error = matrix_2d_sum( &work[ Error_Squared ] );
		} else { // compare to the following layers' delta
			matrix_2d_transpose( &work[ Weight ], &work[ Hidden_T ] );
			matrix_2d_mul( &work[ Hidden_T ], &work[ Little_Delta ], &work[ Hidden_Delta ] );
			matrix_2d_slice_row( &work[ Hidden_Delta ], &work[ Slice ] ); // remove bias row
		}
		bpn_apply_transfer_derivatives( view->transfer_functions[ l ],
			&work[ Layer_Input ], &work[ Derivative ] );
		bpn_layer_matrix( view, Little_Delta, l, num_layers ); // target current layers' little delta!
		matrix_2d_mul_elements( &work[ Slice ], &work[ Derivative ], &work[ Little_Delta ] );
	}
	for( l = 0; l < num_layers; ++l ) { // Compute weight delta
		bpn_prepare_prop( view, num_inputs, num_layers, test_cases, l, n );

		if( 0 == l ) {
			matrix_2d_transpose( input, &work[ Input_T ] );
			matrix_2d_append_ones( &work[ Input_T ], &work[ Layer_Stack ] );
		} else {
			matrix_2d_append_ones( &work[ Layer_Output ], &work[ Layer_Stack ] );
		}
		matrix_3d_from_2d( &work[ Layer_Stack ], &work[ Output_Extended ] );
		matrix_3d_transpose( 2, 0, 1, &work[ Output_Extended ], &work[ Output_T ] );
		matrix_3d_from_2d( &work[ Little_Delta ], &work[ Little_Delta_Extended ] );
		matrix_3d_transpose( 2, 1, 0, &work[ Little_Delta_Extended ], &work[ Little_Delta_T ] );
		matrix_3d_mul( &work[ Output_T ], &work[ Little_Delta_T ], &work[ Product ] );
		matrix_3d_sum_along( 0, &work[ Product ], &work[ Sum ] );
		matrix_2d_mul_scalar_self( &work[ Sum ], network->learning_rate );
		matrix_2d_mul_scalar( &work[ Previous_Weight_Delta ], &work[ Momentum ], network->momentum );
		matrix_2d_add( &work[ Sum ], &work[ Momentum ], &work[ Previous_Weight_Delta ] );
		matrix_2d_sub_self( &work[ Weight ], &work[ Previous_Weight_Delta ] );
	}
	return error;
}

/*
	Persistence
*/
inline
size_t bpn_persistent_size( Network *network ) {
	return NETWORK_HEADER_LEN
		+ ( size_t ) network->view_bytes
		+ ( size_t ) network->weight_bytes;
}

void bpn_copy_data_out( Network *network, View *view, char *buffer ) {
	Info nfo;
	int i, l, len;
	int num_layers = network->num_layers;
	size_t diff = ( char * ) view->data - ( char * ) view->layer_sizes;

	memcpy( buffer, network, NETWORK_HEADER_LEN );
	memcpy( buffer + NETWORK_HEADER_LEN, view->layer_sizes, diff );
	float *p = ( float * )( buffer + NETWORK_HEADER_LEN + diff );

	for( l = 0; l < num_layers; ++l ) {
		bpn_info_get( num_layers, view->info, Weight, l, &nfo );
		len = nfo.shape[ 0 ] * nfo.shape[ 1 ];
		float *d = view->data + nfo.offset;

		for( i = 0; i < len; ++i ) {
			*p++ = *d++;
		}
	}
}
