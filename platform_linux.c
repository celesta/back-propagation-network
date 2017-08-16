#include "network.c"
#include <stdio.h>
#include <time.h>

double platform_get_milliseconds( void ) {
	struct timespec time;
	clock_gettime( CLOCK_MONOTONIC, &time );
	return ( double ) time.tv_sec * 1000.0f + ( double ) time.tv_nsec / 1000000.0f;
}

int bpn_save( const char *file, Network *network, View *view ) {
	int success = 0;
	size_t sz = bpn_persistent_size( network );
	char *buffer = malloc( sz * sizeof( char ) );

	if( buffer ) {
		bpn_copy_data_out( network, view, buffer );
		FILE *fh = fopen( file, "w" );

		if( fh ) {
			size_t written = fwrite( buffer, 1, sz, fh );

			if( sz == written ) {
				success = 1;
			}
			fclose( fh );
		} else {
			success = BPN_No_File_Handle;
		}
		free( buffer );
	} else {
		success = BPN_No_Memory;
	}
	return success;
}

int bpn_load( const char *file, Network *network, View *view ) {
	char *buffer = 0;
	float *weights = 0;
	int a, b, c, d, tc, read, align, num_layers;
	int success = 0;
	size_t offset;
	FILE *fh = fopen( file, "r" );

	if( fh ) {
		read = fread( network, 1, NETWORK_HEADER_LEN, fh );

		if( NETWORK_HEADER_LEN != read ) {
			success = BPN_Invalid_Read;
			goto last;
		}
		tc = network->test_cases;
		a = ( ( tc >= TEST_BLOCK ) && ( 0 == ( tc % TEST_BLOCK ) ) );
		b = ( network->num_layers > 1 );
		c = ( network->num_inputs > 0 );

		if( ! ( a && b && c ) ) {
			success = BPN_Invalid_Size;
			goto last;
		}
		a = ( network->view_bytes > 0 );
		b = ( network->weight_bytes > 0 );
		c = ( network->data_bytes > network->weight_bytes );
		d = ( network->total_bytes > ( network->view_bytes + network->data_bytes ) );

		if( ! ( a && b && c && d ) ) {
			success = BPN_Invalid_Size;
			goto last;
		}
		buffer = calloc( network->total_bytes, sizeof( char ) );

		if( !buffer ) {
			success = BPN_No_Memory;
			goto last;
		}
		num_layers = network->num_layers;
		align = ( 2 * num_layers ) % TEST_BLOCK;

		read = fread( buffer, 1, network->view_bytes, fh );

		if( network->view_bytes != read ) {
			success = BPN_Invalid_Read;
			goto lerr;
		}
		view->layer_sizes = ( int * ) buffer;
		offset = num_layers + align / 2;
		view->transfer_functions = ( int * ) buffer + offset;
		offset += num_layers + align / 2;
		view->info = ( int * ) buffer + offset;
		offset += ( Usage_Max * num_layers ) << 2;
		view->data = ( float * ) buffer + offset;
		offset += ( network->data_bytes / sizeof( float ) );
		view->error_squared = ( float * ) buffer + offset;
		offset += tc;
		view->input_T = ( float * ) buffer + offset;
		offset += tc * network->num_inputs;

		weights = malloc( network->weight_bytes * sizeof( char ) );

		if( !weights ) {
			success = BPN_No_Memory;
			goto lerr;
		}
		read = fread( weights, 1, network->weight_bytes, fh );

		if( network->weight_bytes != read ) {
			free( weights );
			success = BPN_Invalid_Read;
			goto lerr;
		}
		bpn_set_weights( view, num_layers, weights );
		network->storage = buffer;
		success = 1;
		free( weights );
		goto last;
	} else {
		return BPN_No_File_Handle;
	}
lerr:
	if( buffer ) {
		free( buffer );
	}
last:
	fclose( fh );
	return success;
}

void bpn_free( Network *network ) {
	if( network->storage ) {
		free( network->storage );
		network->storage = 0;
	}
}
