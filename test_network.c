/*
	Try to train a network so it can act as an XOR gate.
	On success save network to file, load and run it.
*/
#include "platform_linux.c"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void matrix_2d_print( int width, int height, const float *m );

const char *file = "/tmp/bp_xor";

#ifdef DEBUG_VERSION
void bpn_print( Network *network, View *view, int header_only );
void matrix_3d_print( int width, int height, int depth, const float *m );
const char *usage_to_string( int usage );
#endif

int main( int argc, char *argv[ ] ) {
	Network network;
	View view;
	Matrix m_input, m_output, m_target;
	float error, *mem, *input, *output, *target;
	float error_margin = 1E-5;
	int i, success, saved, step, mod, num_layers;
	int max_iterations = 10000;
	int test_cases = TEST_BLOCK; // must be multiple of TEST_BLOCK to work with Simd
	int num_inputs = 2;
	int num_outputs = 1;
	int hidden_size = 2;
	int transfer_functions[ ] = { TF_Gaussian, TF_Linear };
	int layer_sizes[ ] = { num_inputs, hidden_size, num_outputs };
	// Data could come from a file so imitate check...
	num_layers = sizeof( layer_sizes ) / sizeof( layer_sizes[ 0 ] ) - 1;
	assert( num_layers == sizeof( transfer_functions ) / sizeof( transfer_functions[ 0 ] ) );
	assert( 0 == ( test_cases % TEST_BLOCK ) );

	mem = malloc( test_cases * ( num_inputs + 2 * num_outputs ) * sizeof( float ) );

	if( !mem ) {
		printf( "Could not allocate basic storage - aborting!\n" );
		return -1;
	}

	input = mem;
	size_t offset = test_cases * num_inputs;
	output = mem + offset;
	offset += test_cases * num_outputs;
	target = mem + offset;

	int input_shape[ ] = { test_cases, num_inputs, 0 };
	int output_shape[ ] = { test_cases, num_outputs, 0 };
	int target_shape[ ] = { test_cases, num_outputs, 0 };

	bpn_general_matrix( input, input_shape, &m_input );
	bpn_general_matrix( output, output_shape, &m_output );
	bpn_general_matrix( target, target_shape, &m_target );

	matrix_2d_set( input, num_inputs, 0, 0, 0.0f );
	matrix_2d_set( input, num_inputs, 0, 1, 0.0f );
	matrix_2d_set( input, num_inputs, 1, 0, 0.0f );
	matrix_2d_set( input, num_inputs, 1, 1, 1.0f );
	matrix_2d_set( input, num_inputs, 2, 0, 1.0f );
	matrix_2d_set( input, num_inputs, 2, 1, 0.0f );
	matrix_2d_set( input, num_inputs, 3, 0, 1.0f );
	matrix_2d_set( input, num_inputs, 3, 1, 1.0f );

	matrix_2d_set( target, num_outputs, 0, 0, 0.0f );
	matrix_2d_set( target, num_outputs, 1, 0, 1.0f );
	matrix_2d_set( target, num_outputs, 2, 0, 1.0f );
	matrix_2d_set( target, num_outputs, 3, 0, 0.0f );

	memset( &network.name, 0, SIGNATURE_LEN * sizeof( char ) );
	snprintf( network.name, SIGNATURE_LEN * sizeof( char ), "%s", "XOR gate" );
	network.learning_rate = 0.2f;
	network.momentum = 0.5f;
	network.test_cases = test_cases;
	network.num_layers = num_layers;
	network.num_inputs = num_inputs;
	network.flags = 0;

	srand( time( 0 ) /*1001*/ );
	success = bpn_allocate( &network, &view, layer_sizes, transfer_functions );

	if( 0 != success ) {
		printf( "Could not allocate network storage - aborting!\n" );
		return -1;
	}

	mod = ( int )( ( float ) max_iterations / 10.0f );
	step = 0;
	double t_elapsed, t_start, t_end, t_sum = 0.0f;

	printf( "Network will train for %d iterations with an error margin of %f\n",
		max_iterations, error_margin );

	for( i = 0; i < max_iterations; ++i ) {
		t_start = platform_get_milliseconds( );
		error = bpn_train( &network, &view, &m_input, &m_target );
		t_end = platform_get_milliseconds( );
		t_elapsed = t_end - t_start;
		t_sum += t_elapsed;
		step = i % mod;

		if( ( 0 == step ) && ( 0 != i ) ) {
			printf( "Iteration: %6d\tTime last: %.6f ms\tError: %f\n",
				i, t_elapsed, error );
		}
		if( error < error_margin ){
			success = 1;
			break;
		}
	}
	network.flags = Training_Done;
	saved = 0;

	if( success ) {
		printf( "Minimum error of %f reached at iteration %d, time: %.3f ms.\n",
			error, i, t_sum );
		network.flags |= Training_Success;
		bpn_run( &network, &view, &m_input, &m_output );
		printf( "Output:\n" );
		matrix_2d_print( test_cases, num_outputs, output );
		saved = bpn_save( file, &network, &view );

		if( !saved ) {
			printf( "Could not save network to file %s!\n\tError was: %d\n", file, saved );
		} else {
			printf( "Network saved to file %s.\n", file );
		}
	} else {
		printf( "Training failed.\tTime: %.3f ms,\tLast error: %f\n", t_sum, error );
	}
	bpn_free( &network );
	memset( &network, 0, sizeof( Network ) );
	memset( output, 0, test_cases * num_outputs * sizeof( float ) );

	if( saved ) {
		success = bpn_load( file, &network, &view );

		if( 1 == success ) {
			printf( "Network loaded from file %s.\n", file );
#ifdef DEBUG_VERSION
			bpn_print( &network, &view, 1 );
#endif
			bpn_run( &network, &view, &m_input, &m_output );
			printf( "Output:\n" );
			matrix_2d_print( test_cases, num_outputs, output );
			bpn_free( &network );
		} else {
			printf( "Could not load network from file %s!\n\tError was: %d\n", file, success );
		}
	}
	free( mem );
	return 0;
}

void matrix_2d_print( int width, int height, const float *m ) {
	int i, j;

	printf( "\t[" );
	for( i = 0; i < width; ++i ) {
		printf( "[" );

		for( j = 0; j < height; ++j ) {
			float f = matrix_2d_get( m, height, i, j );
			printf( "%f", f );
			if( j < ( height - 1 ) ) {
				printf( ", " );
			}
		}
		if( i < ( width - 1 ) ) {
			printf( "],\n\t " );
		} else {
			printf( "]" );
		}
	}
	printf( "]\n" );
}

#ifdef DEBUG_VERSION
void bpn_print( Network *network, View *view, int header_only ) {
	int i, j;
	int num_layers = network->num_layers;
	const char *usage;

	printf( "\nNetwork: %s\n", network->name );
	printf( "\tLearning rate:  %.6f\n", network->learning_rate );
	printf( "\tMomentum:       %.6f\n", network->momentum );
	printf( "\tTest cases:     %8d\n", network->test_cases );
	printf( "\tNo. of layers:  %8d\n", network->num_layers );
	printf( "\tNo. of inputs:  %8d\n", network->num_inputs );
	printf( "\tFlags:          %8X\n", network->flags );
	printf( "\tBytes view:     %8d\n", network->view_bytes );
	printf( "\tBytes weight:   %8d\n", network->weight_bytes );
	printf( "\tBytes data:     %8d\n", network->data_bytes );
	printf( "\tBytes total:    %8d\n", network->total_bytes );
	printf( "\n" );

	if( header_only ) {
		return;
	}

	for( j = 0; j < num_layers; ++j ) {
		printf( "Layer %d\n", j );

		for( i = 0; i < Usage_Max; ++i ) {
			Info nfo;
			bpn_info_get( num_layers, view->info, i, j, &nfo );
			usage = usage_to_string( i );

			if( i < Output_Extended ) {
				printf( "%s (%d,%d), offset %u\n", usage, nfo.shape[ 0 ], nfo.shape[ 1 ], nfo.offset );
				matrix_2d_print( nfo.shape[ 0 ], nfo.shape[ 1 ], ( view->data + nfo.offset ) );
			} else {
				printf( "%s (%d,%d,%d), offset %u\n", usage, nfo.shape[ 0 ], nfo.shape[ 1 ], nfo.shape[ 2 ], nfo.offset );
				matrix_3d_print( nfo.shape[ 0 ], nfo.shape[ 1 ], nfo.shape[ 2 ], ( view->data + nfo.offset ) );
			}
		}
	}
}

void matrix_3d_print( int width, int height, int depth, const float *m ) {
	int i, j, k;

	for( i = 0; i < width; ++i ) {
		printf( "\t[" );

		for( j = 0; j < height; ++j ) {
			printf( "[" );

			for( k = 0; k < depth; ++k ) {
				float f = matrix_3d_get( m, height, depth, i, j, k );
				printf( "%f", f );
				if( k < ( depth - 1 ) ) {
					printf( ", " );
				}
			}
			if( j < ( height - 1 ) ) {
				printf( "],\n\t " );
			} else {
				printf( "]" );
			}
		}
		if( i < ( width - 1 ) ) {
			printf( "],\n" );
		} else {
			printf( "]\n" );
		}
	}
}

const char *usage_to_string( int usage ) {
	const char *name = 0;

	switch( usage ) {
		case Weight:
			name = "Weight";
			break;
		case Previous_Weight_Delta:
			name = "Previous weight delta";
			break;
		case Momentum:
			name = "Momentum";
			break;
		case Sum:
			name = "Sum";
			break;
		case Layer_Input:
			name = "Layer input";
			break;
		case Layer_Output:
			name = "Layer output";
			break;
		case Layer_Stack:
			name = "Layer stack";
			break;
		case Little_Delta:
			name = "Little delta";
			break;
		case Derivative:
			name = "Derivative";
			break;
		case Slice:
			name = "Slice";
			break;
		case Hidden_Delta:
			name = "Hidden delta";
			break;
		case Hidden_T:
			name = "Hidden transpose";
			break;
		case Output_Extended:
			name = "Output extended";
			break;
		case Output_T:
			name = "Output transposed";
			break;
		case Little_Delta_Extended:
			name = "Little delta extended";
			break;
		case Little_Delta_T:
			name = "Little delta transposed";
			break;
		case Product:
			name = "Product";
			break;
	};
	return name;
}
#endif /* DEBUG_VERSION */
