/****************************************************************************
 *
 * MODULE:  penelope.hpp
 * PURPOSE: penelope nif helpers
 *
 ***************************************************************************/
#ifndef __PENELOPE_HPP
#define __PENELOPE_HPP
/*-------------------[       Pre Include Defines       ]-------------------*/
/*-------------------[      Library Include Files      ]-------------------*/
#include <stdio.h>
#include <string.h>
#include <erl_nif.h>
/*-------------------[      Project Include Files      ]-------------------*/
/*-------------------[      Macros/Constants/Types     ]-------------------*/
#define CHECK(result, ...)                                                 \
   ({                                                                      \
      auto r = (result);                                                   \
      if (!r)                                                              \
         throw NifError(__VA_ARGS__);                                      \
      r;                                                                   \
   })
#define CHECKALLOC(result)                                                 \
   CHECK(result, "alloc_failed");
/*-------------------[             Classes             ]-------------------*/
/*-----------< CLASS: NifError >---------------------------------------------
// Purpose:    simple nif exception class
---------------------------------------------------------------------------*/
class NifError {
public:
   NifError () {
      strncpy(_code, "unknown", sizeof(_code) - 1);
      _code[sizeof(_code) - 1] = 0;
   }
   NifError (const char* code) {
      strncpy(_code, code, sizeof(_code) - 1);
      _code[sizeof(_code) - 1] = 0;
   }
   const char* code () const {
      return _code;
   }
   ERL_NIF_TERM to_term (ErlNifEnv* env, const char* reason = NULL) const {
      return enif_raise_exception(
         env,
         enif_make_tuple2(
            env,
            enif_make_atom(env, code()),
            reason ? enif_make_string(env, reason, ERL_NIF_LATIN1) :
                     enif_make_atom(env, "nil")));
   }
private:
   char _code[128 + 1];
};
/*-------------------[        Global Variables         ]-------------------*/
/*-------------------[        Global Prototypes        ]-------------------*/
template<typename T> inline T* nif_alloc (int count = 1) {
   T* t = (T*)calloc(count, sizeof(T));
   CHECKALLOC(t);
   return t;
}
template<typename T> inline T* nif_clone (const T* source, int count = 1) {
   T* t = nif_alloc<T>(count);
   memcpy(t, source, count * sizeof(T));
   return t;
}
inline void nif_free (void* t) {
   if (t)
      free(t);
}
#endif // __PENELOPE_HPP
