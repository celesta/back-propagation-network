/*
	Back propagation network - gradient descent + momentum
*/
#ifndef H_NETWORK
#define H_NETWORK

#define TEST_BLOCK		(4)
#define SIGNATURE_LEN	(16)

enum { // Usage types of matrices
	Weight,							// Twelve 2d matrices per layer
	Previous_Weight_Delta,
	Momentum,
	Sum,
	Layer_Input,
	Layer_Output,
	Little_Delta,
	Derivative,
	Slice,
	Layer_Stack,
	Hidden_Delta,
	Hidden_T,
	Output_Extended,				// Five 3d matrices per layer
	Output_T,
	Little_Delta_Extended,
	Little_Delta_T,
	Product,
	Usage_Max,
	Error_Squared = Usage_Max,		// Two special 2d matrices
	Input_T
};

enum { // Flags
	Training_Done		= 0x0001,
	Training_Success	= 0x0002
};

enum { // Error indicators
	BPN_No_File_Handle	= -1,
	BPN_No_Memory		= -2,
	BPN_Invalid_Read	= -3,
	BPN_Invalid_Write	= -4,
	BPN_Invalid_Size	= -5
};

typedef struct {
	int shape[ 3 ];			// width, height and depth
	float *data;
} Matrix;					// Note: aligns to 24 bytes, not 12 + 8

typedef struct {
	int shape[ 3 ];			// width, height and depth
	int offset;				// bytes into data of struct View
} Info;						// 16 bytes

typedef struct {
	int *layer_sizes;			// maybe pad allocated storage here
	int *transfer_functions;	// and there so that actual data
	int *info;					// from here on will be aligned at a 16 byte boundary
	float *data;
	float *error_squared;
	float *input_T;
	Matrix work[ Usage_Max + 2 ];
} View;

typedef struct {
	char name[ SIGNATURE_LEN ];
	float learning_rate;
	float momentum;
	int test_cases;
	int num_layers;
	int num_inputs;
	int flags;					// training done etc.
	unsigned int view_bytes;	// layer_sizes, transfer funcs and info
	unsigned int weight_bytes;
	unsigned int data_bytes;
	unsigned int total_bytes;
	char *storage;
} Network;						// 64 bytes hopefully

/*
	Note: since a shape array is used for 2d and 3d there's a little memory over-head:
		for info data:
			12 * sizeof int bytes per layer
		for work data:
			12 * sizeof int
			+ struct Matrix gets aligned to 24 bytes (real 20):
			Usage_Max * sizeof int

	eg. for 2 layers: ( 2 * 12 + 12 + 17 ) * 4 = 53 * 4 = 212 bytes
	Accepted, since number of layers is usually very small.
*/

void bpn_general_matrix( float *data, const int *shape, Matrix *out );

int bpn_allocate( Network *network, View *view, int *layer_sizes, int *transfers );

float bpn_train( Network *network, View *view, const Matrix *input, const Matrix *target );

void bpn_run( Network *network, View *view, const Matrix *input, Matrix *output );

int bpn_save( const char *file, Network *network, View *view );

int bpn_load( const char *file, Network *network, View *view );

void bpn_free( Network *network );

#endif /* H_NETWORK */
