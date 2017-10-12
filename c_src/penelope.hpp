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
   {                                                                       \
      if (!(result))                                                       \
         throw NifError(__VA_ARGS__);                                      \
   }
#define CHECKALLOC(result)                                                 \
   CHECK(result, "alloc_failed");
/*-------------------[             Classes             ]-------------------*/
/*-----------< CLASS: NifError >---------------------------------------------
// Purpose:    simple nif exception class
---------------------------------------------------------------------------*/
class NifError {
public:
   NifError () {
      _code = "unknown";
   }
   NifError (const char* code) {
      _code = code;
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
   const char* _code;
};
/*-------------------[        Global Variables         ]-------------------*/
/*-------------------[        Global Prototypes        ]-------------------*/
template<typename T> inline T* nif_alloc (int count = 1) {
   T* t = (T*)malloc(count * sizeof(T));
   CHECKALLOC(t);
   memset(t, 0, count * sizeof(T));
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
