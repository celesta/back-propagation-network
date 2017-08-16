#ifndef H_MACROS
#define H_MACROS

#define min(a, b)			(((a) < (b)) ? (a) : (b))
#define max(a, b)			(((a) > (b)) ? (a) : (b))
#define clamp(v, a, b)		(max(min((a), (b)), (v)))

#endif /* H_MACROS */
