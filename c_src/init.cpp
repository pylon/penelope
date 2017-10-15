/****************************************************************************
 *
 * MODULE:  init.cpp
 * PURPOSE: nif initialization and export
 *
 ***************************************************************************/
/*-------------------[       Pre Include Defines       ]-------------------*/
/*-------------------[      Library Include Files      ]-------------------*/
/*-------------------[      Project Include Files      ]-------------------*/
#include "penelope.hpp"
/*-------------------[      Macros/Constants/Types     ]-------------------*/
#define DECLARE_NIF(name)                                                  \
   extern ERL_NIF_TERM nif_##name (ErlNifEnv*, int, const ERL_NIF_TERM[])
#define EXPORT_NIF(name, args, ...)                                        \
   (ErlNifFunc){ #name, args, nif_##name, __VA_ARGS__ }
/*-------------------[        Global Variables         ]-------------------*/
/*-------------------[        Global Prototypes        ]-------------------*/
extern int nif_blas_init (ErlNifEnv* env);
extern int nif_svm_init  (ErlNifEnv* env);
/*-------------------[        Module Variables         ]-------------------*/
/*-------------------[        Module Prototypes        ]-------------------*/
DECLARE_NIF(blas_sscal);
DECLARE_NIF(blas_saxpy);
DECLARE_NIF(svm_train);
DECLARE_NIF(svm_export);
DECLARE_NIF(svm_compile);
DECLARE_NIF(svm_predict_class);
DECLARE_NIF(svm_predict_probability);
/*-------------------[         Implementation          ]-------------------*/
// nif function table
static ErlNifFunc nif_map[] = {
   EXPORT_NIF(blas_sscal, 2),
   EXPORT_NIF(blas_saxpy, 3),
   EXPORT_NIF(svm_train, 3, ERL_NIF_DIRTY_JOB_CPU_BOUND),
   EXPORT_NIF(svm_export, 1),
   EXPORT_NIF(svm_compile, 1),
   EXPORT_NIF(svm_predict_class, 2),
   EXPORT_NIF(svm_predict_probability, 2)
};
/*-----------< FUNCTION: nif_loaded >----------------------------------------
// Purpose:    nif onload callback
// Parameters: env       - erlang environment
//             priv_data - return private state via here
//             state     - nif load parameter
// Returns:    1 if successful
//             0 otherwise
---------------------------------------------------------------------------*/
static int nif_loaded (ErlNifEnv* env, void** priv_data, ERL_NIF_TERM state)
{
   *priv_data = NULL;
   if (!nif_blas_init(env))
      return 1;
   if (!nif_svm_init(env))
      return 2;
   return 0;
}
// nif entry point
ERL_NIF_INIT(
   Elixir.Penelope.NIF,
   nif_map,
   &nif_loaded,
   NULL,
   NULL,
   NULL);
