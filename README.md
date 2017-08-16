
	Back propagation network, gradient descent & momentum, written in C.

	Based on and inspired by the tutorials of Ryan Harris.
	Thanks to him, who explains neural networks and the workings
	of the back propagation algorithm in such an excellent manner,
	i am able to proudly present his python version in C.

	Here are links to his contributions:

		(https://www.youtube.com/channel/UCRAmB5K-2GLvtaXcH9GCy-A)
		(https://github.com/TheFellow)

	Thanks to Julien Pommier,
		whos SIMD implementation of sin, cos, exp and log i intend
		on using for the vectorized version.

	Thanks to everybody.

	The test program trains a network for computing the XOR function,
	then saves, loads and runs it again.

	To compile the debug version i use:

		gcc -Wall -O2 -ggdb -o network test_network.c -lm

	or for the optimized and stripped version without assertions:

		gcc -Wall -O2 -s -DNDEBUG -o network test_network.c -lm

	Developed on x86_64 linux, never ran it on anything else but
	should be easy to port (see platform_linux.c).

	The network:

		The number of provided test cases must be a multiple of 4 to
		allow the use of vectorized transfer functions in the future.
		Right now they just pretend to be, although the data layout
		is prepared for aligned access (16 bytes).

	I followed Ryans explanations as closely as i could, but lacking
	the mighty numpy i had some, not unpleasant, fuzz with matrices.
	Also, to get an at least somewhat efficient network, it allocates
	all the memory needed for data and computations during run and
	training phases at once. Other than that i did not try to optimize
	the program, and to be frank, i'm eager to know how improvements
	could be made.

	Mind you, although inwardly i contest it, the code might have bugs.
	So, besides wanting to show off, i publish this also to get	some,
	any feedback really, which i otherwise could not.

	Coding this was a step to my real goal, which is to build
	evolutionary networks and run them simultaneousely.
	Since those are, in my understanding, ever altering in shape
	(some might even become fully connected by mutation) it probably
	deserves a different approach in respect to the data layout.
	Ofcourse i would much prefer if i could just expand on my
	existing implementation, but i haven't thought too hard about it.
	Like, if i just could set a weight to ~0 to mean no link existed
	and everything would still work ^^.
