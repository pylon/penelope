/****************************************************************************
 *
 * MODULE:  blas.cpp
 * PURPOSE: nifs for basic linear algebra subprograms (CBLAS)
 *
 ***************************************************************************/
/*-------------------[       Pre Include Defines       ]-------------------*/
/*-------------------[      Library Include Files      ]-------------------*/
#ifdef __APPLE__
#  include <Accelerate/Accelerate.h>
#else
#  include <cblas.h>
#endif
/*-------------------[      Project Include Files      ]-------------------*/
#include "penelope.hpp"
/*-------------------[      Macros/Constants/Types     ]-------------------*/
/*-------------------[        Global Variables         ]-------------------*/
/*-------------------[        Global Prototypes        ]-------------------*/
/*-------------------[        Module Variables         ]-------------------*/
/*-------------------[        Module Prototypes        ]-------------------*/
/*-------------------[         Implementation          ]-------------------*/
/*-----------< FUNCTION: nif_blas_init >-------------------------------------
// Purpose:    blas module initialization
// Parameters: env - erlang environment
// Returns:    1 if successful
//             0 otherwise
---------------------------------------------------------------------------*/
int nif_blas_init (ErlNifEnv* env)
{
   return 1;
}
/*-----------< FUNCTION: nif_blas_sscal >------------------------------------
// Purpose:    BLAS sscal wrapper
//             computes y = ax
// Parameters: a - scalar to multiply (float)
//             x - vector to multiply (binary float vector)
// Returns:    result of ax (binary float vector)
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_blas_sscal(
   ErlNifEnv*         env,
   int                argc,
   const ERL_NIF_TERM argv[])
{
   double a;
   ErlNifBinary x, r;
   // validate parameters
   if (!enif_get_double(env, argv[0], &a))
      return enif_make_badarg(env);
   if (!enif_inspect_binary(env, argv[1], &x))
      return enif_make_badarg(env);
   // copy x as the result, which blas will overwrite
   if (!enif_alloc_binary(x.size, &r))
      return enif_raise_exception(env, enif_make_atom(env, "alloc_failed"));
   memcpy(r.data, x.data, r.size);
   // perform the vectorized computation
   cblas_sscal(
      r.size / sizeof(float),
      a,
      (float*)r.data,
      1);
   return enif_make_binary(env, &r);
}
/*-----------< FUNCTION: nif_blas_saxpy >------------------------------------
// Purpose:    BLAS saxpy wrapper
//             computes z = ax + y
// Parameters: a - scalar to multiply (float)
//             x - vector to multiply (binary float vector)
//             y - vector to add (binary float vector)
// Returns:    result of ax + y (binary float vector)
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_blas_saxpy(
   ErlNifEnv*         env,
   int                argc,
   const ERL_NIF_TERM argv[])
{
   double a;
   ErlNifBinary x, y, r;
   // validate parameters
   if (!enif_get_double(env, argv[0], &a))
      return enif_make_badarg(env);
   if (!enif_inspect_binary(env, argv[1], &x))
      return enif_make_badarg(env);
   if (!enif_inspect_binary(env, argv[2], &y))
      return enif_make_badarg(env);
   if (x.size != y.size)
      return enif_make_badarg(env);
   // copy y as the result, which blas will overwrite
   if (!enif_alloc_binary(y.size, &r))
      return enif_raise_exception(env, enif_make_atom(env, "alloc_failed"));
   memcpy(r.data, y.data, r.size);
   // perform the vectorized computation
   cblas_saxpy(
      x.size / sizeof(float),
      a,
      (float*)x.data,
      1,
      (float*)r.data,
      1);
   return enif_make_binary(env, &r);
}
