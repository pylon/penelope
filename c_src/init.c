/****************************************************************************
 *
 * MODULE:  init.c
 * PURPOSE: nif initialization and export
 *
 ***************************************************************************/
/*-------------------[       Pre Include Defines       ]-------------------*/
/*-------------------[      Library Include Files      ]-------------------*/
#include <erl_nif.h>
/*-------------------[      Project Include Files      ]-------------------*/
/*-------------------[      Macros/Constants/Types     ]-------------------*/
#define DECLARE_NIF(name, args)                                            \
   extern ERL_NIF_TERM nif_##name (ErlNifEnv*, int, const ERL_NIF_TERM[]); \
   static const int __##name##__args = args;
#define EXPORT_NIF(name)                                                   \
   { #name, __##name##__args, nif_##name }
/*-------------------[        Global Variables         ]-------------------*/
/*-------------------[        Global Prototypes        ]-------------------*/
/*-------------------[        Module Variables         ]-------------------*/
/*-------------------[        Module Prototypes        ]-------------------*/
DECLARE_NIF(blas_sscal, 2);
DECLARE_NIF(blas_saxpy, 3);
/*-------------------[         Implementation          ]-------------------*/
// nif function table
static ErlNifFunc nif_map[] = {
   EXPORT_NIF(blas_sscal),
   EXPORT_NIF(blas_saxpy)
};
// nif entry point
ERL_NIF_INIT(
   Elixir.Penelope.NIF,
   nif_map,
   NULL,
   NULL,
   NULL,
   NULL);
