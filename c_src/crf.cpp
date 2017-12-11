/****************************************************************************
 *
 * MODULE:  crf.cpp
 * PURPOSE: nifs for crfsuite
 *
 * see http://www.chokkan.org/software/crfsuite/ for details
 *
 ***************************************************************************/
/*-------------------[       Pre Include Defines       ]-------------------*/
/*-------------------[      Library Include Files      ]-------------------*/
#include <math.h>
#include <unistd.h>
/*-------------------[      Project Include Files      ]-------------------*/
#include "deps/crfsuite/include/crfsuite.h"
#include "penelope.hpp"
/*-------------------[      Macros/Constants/Types     ]-------------------*/
typedef struct tagCrfModel {
   char              path[PATH_MAX + 1];
   crfsuite_model_t* crf;
} CRF_MODEL;
/*-------------------[        Global Variables         ]-------------------*/
/*-------------------[        Global Prototypes        ]-------------------*/
/*-------------------[        Module Variables         ]-------------------*/
static ErlNifResourceType* g_model_type = NULL;
/*-------------------[        Module Prototypes        ]-------------------*/
static void nif_destruct_model (
   ErlNifEnv* env,
   void*      object);
static crfsuite_trainer_t* erl2crf_trainer (
   ErlNifEnv*          env,
   const ERL_NIF_TERM& options);
static void erl2crf_params (
   ErlNifEnv*          env,
   const ERL_NIF_TERM& options,
   crfsuite_trainer_t* trainer);
static void erl2crf_param_bool(
   ErlNifEnv*          erl_env,
   const ERL_NIF_TERM& erl_params,
   const char*         erl_name,
   crfsuite_params_t*  crf_params,
   const char*         crf_name = NULL);
static void erl2crf_param_int(
   ErlNifEnv*          erl_env,
   const ERL_NIF_TERM& erl_params,
   const char*         erl_name,
   crfsuite_params_t*  crf_params,
   const char*         crf_name = NULL);
static void erl2crf_param_float(
   ErlNifEnv*          erl_env,
   const ERL_NIF_TERM& erl_params,
   const char*         erl_name,
   crfsuite_params_t*  crf_params,
   const char*         crf_name = NULL);
static void erl2crf_param_string(
   ErlNifEnv*          erl_env,
   const ERL_NIF_TERM& erl_params,
   const char*         erl_name,
   crfsuite_params_t*  crf_params,
   const char*         crf_name = NULL);
static void erl2crf_train_data(
   ErlNifEnv*       erl_env,
   ERL_NIF_TERM     x,
   ERL_NIF_TERM     y,
   crfsuite_data_t* data);
static void erl2crf_train_instance(
   ErlNifEnv*             erl_env,
   ERL_NIF_TERM           x_i,
   ERL_NIF_TERM           y_i,
   crfsuite_dictionary_t* crf_attrs,
   crfsuite_dictionary_t* crf_labels,
   crfsuite_instance_t*   crf_instance);
static void erl2crf_predict_instance(
   ErlNifEnv*             erl_env,
   ERL_NIF_TERM           x_i,
   crfsuite_dictionary_t* crf_attrs,
   crfsuite_instance_t*   crf_instance);
static void erl2crf_features(
   ErlNifEnv*             erl_env,
   const ERL_NIF_TERM&    erl_features,
   crfsuite_dictionary_t* crf_attrs,
   crfsuite_instance_t*   crf_instance,
   int                    index,
   bool                   training);
static void erl2crf_feature(
   ErlNifEnv*             erl_env,
   ERL_NIF_TERM&          erl_key,
   ERL_NIF_TERM&          erl_value,
   crfsuite_dictionary_t* crf_attrs,
   crfsuite_item_t&       crf_item,
   bool                   training);
static void erl2crf_label(
   ErlNifEnv*             erl_env,
   const ERL_NIF_TERM&    erl_label,
   crfsuite_dictionary_t* crf_labels,
   crfsuite_instance_t*   crf_instance,
   int                    index);
static void erl2crf_free_train_data (
   crfsuite_data_t* data);
static void erl2crf_free_model (
   CRF_MODEL* model);
static int crf_create_file(
   char* path);
static ERL_NIF_TERM crf2erl_labels(
   ErlNifEnv*             erl_env,
   crfsuite_dictionary_t* crf_labels,
   int*                   crf_path,
   int                    n);
/*-------------------[         Implementation          ]-------------------*/
/*-----------< FUNCTION: nif_crf_init >--------------------------------------
// Purpose:    crf module initialization
// Parameters: env - erlang environment
// Returns:    1 if successful
//             0 otherwise
---------------------------------------------------------------------------*/
int nif_crf_init (ErlNifEnv* env)
{
   // register the model resource type,
   // which holds trained CRF model instances
   ErlNifResourceFlags flags = ERL_NIF_RT_CREATE;
   g_model_type = enif_open_resource_type(
      env,
      NULL,
      "crf_model",
      &nif_destruct_model,
      flags,
      &flags);
   if (!g_model_type)
      return 0;
   return 1;
}
/*-----------< FUNCTION: nif_crf_train >-------------------------------------
// Purpose:    trains a CRF model
// Parameters: x      - list of list of features (map)
//             y      - list of list of labels (string)
//             params - map of CRF parameters
// Returns:    reference to a trained CRF model resource
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_crf_train (
   ErlNifEnv*         env,
   int                argc,
   const ERL_NIF_TERM argv[])
{
   // validate parameters
   if (!enif_is_list(env, argv[0]))
      return enif_make_badarg(env);
   if (!enif_is_list(env, argv[1]))
      return enif_make_badarg(env);
   if (!enif_is_map(env, argv[2]))
      return enif_make_badarg(env);
   // train the CRF model
   crfsuite_trainer_t* trainer  = NULL;
   CRF_MODEL* model = NULL;
   ERL_NIF_TERM result;
   try {
      // allocate and configure the model trainer
      trainer = erl2crf_trainer(env, argv[2]);
      erl2crf_params(env, argv[2], trainer);
      // allocate a new model instance and file
      model = nif_alloc<CRF_MODEL>();
      close(crf_create_file(model->path));
      // build the training data structure and train the model
      crfsuite_data_t train_data;
      try {
         erl2crf_train_data(env, argv[0], argv[1], &train_data);
         CHECK(trainer->train(trainer, &train_data, model->path, -1) == 0,
            "train_failed");
         erl2crf_free_train_data(&train_data);
      } catch (NifError& e) {
         erl2crf_free_train_data(&train_data);
         throw;
      }
      // load the CRF model from the model file
      CHECK(crfsuite_create_instance_from_file(
            model->path,
            (void**)&model->crf) == 0,
         "load_failed");
      // create an erlang resource for the model
      CRF_MODEL** resource = (CRF_MODEL**)CHECKALLOC(enif_alloc_resource(
         g_model_type,
         sizeof(CRF_MODEL*)));
      *resource = model;
      // relinquish the model resource to erlang
      result = enif_make_resource(env, resource);
      enif_release_resource(resource);
   } catch (NifError& e) {
      if (model != NULL)
         erl2crf_free_model(model);
      result = e.to_term(env);
   }
   // clean up
   if (trainer != NULL)
      trainer->release(trainer);
   return result;
}
/*-----------< FUNCTION: nif_crf_export >------------------------------------
// Purpose:    extracts model parameters from a CRF resource,
//             which is useful for persisting a model externally
// Parameters: model - erlang resource wrapping the trained model
// Returns:    model parameter map
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_crf_export (
   ErlNifEnv*         env,
   int                argc,
   const ERL_NIF_TERM argv[])
{
   // validate parameters
   if (!enif_is_ref(env, argv[0]))
      return enif_make_badarg(env);
   // extract the model resource
   CRF_MODEL** resource = NULL;
   if (!enif_get_resource(env, argv[0], g_model_type, (void**)&resource))
      return enif_make_badarg(env);
   // convert the resource to a map
   ErlNifBinary buffer; memset(&buffer, 0, sizeof(buffer));
   FILE* file = NULL;
   long int size = 0;
   ERL_NIF_TERM result;
   try {
      // load the model data into a buffer
      file = CHECK(fopen((*resource)->path, "rb"), "load_failed");
      CHECK(fseek(file, 0, SEEK_END) == 0, "load_failed");
      size = ftell(file);
      CHECK(size != -1, "load_failed");
      CHECK(fseek(file, 0, SEEK_SET) == 0, "load_failed");
      CHECKALLOC(enif_alloc_binary(size, &buffer));
      CHECK(fread(buffer.data, 1, size, file) == size, "load_failed");
      // add the model buffer to a map
      ERL_NIF_TERM key   = enif_make_atom(env, "model");
      ERL_NIF_TERM value = enif_make_binary(env, &buffer);
      result = enif_make_new_map(env);
      CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   } catch (NifError& e) {
      if (buffer.data)
         enif_release_binary(&buffer);
      result = e.to_term(env);
   }
   // clean up
   if (file != NULL)
      fclose(file);
   return result;
}
/*-----------< FUNCTION: nif_crf_compile >-----------------------------------
// Purpose:    converts the map representation of a model to the
//             native CRF model structure
// Parameters: model - map containing model parameters
// Returns:    reference to a trained CRF model resource
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_crf_compile (
   ErlNifEnv*         env,
   int                argc,
   const ERL_NIF_TERM argv[])
{
   // validate parameters
   ERL_NIF_TERM key = enif_make_atom(env, "model");
   ERL_NIF_TERM value;
   ErlNifBinary buffer;
   if (!enif_get_map_value(env, argv[0], key, &value))
      return enif_make_badarg(env);
   if (!enif_inspect_binary(env, value, &buffer))
      return enif_make_badarg(env);
   // compile the model parameters
   FILE* file = NULL;
   CRF_MODEL* model = NULL;
   ERL_NIF_TERM result;
   try {
      // write the model buffer to a new file
      model = nif_alloc<CRF_MODEL>();
      file = CHECK(fdopen(crf_create_file(model->path), "wb"),
         "store_failed");
      CHECK(fwrite(buffer.data, 1, buffer.size, file) == buffer.size,
         "store_failed");
      fflush(file);
      // load the CRF model from the model file
      CHECK(crfsuite_create_instance_from_file(
            model->path,
            (void**)&model->crf) == 0,
         "load_failed");
      // create an erlang resource for the model
      CRF_MODEL** resource = (CRF_MODEL**)CHECKALLOC(enif_alloc_resource(
         g_model_type,
         sizeof(CRF_MODEL*)));
      *resource = model;
      // relinquish the model resource to erlang
      result = enif_make_resource(env, resource);
      enif_release_resource(resource);
   } catch (NifError& e) {
      if (model != NULL)
         erl2crf_free_model(model);
      result = e.to_term(env);
   }
   // clean up
   if (file != NULL)
      fclose(file);
   return result;
}
/*-----------< FUNCTION: nif_crf_predict >-----------------------------------
// Purpose:    predicts a sequence of tags from a sequence of features
// Parameters: model - reference to the trained CRF model
//             x     - feature sequence (list) to predict
// Returns:    a tuple containing the predicted tag sequence (list) and the
//             probability of sequence
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_crf_predict (
   ErlNifEnv* env,
   int        argc,
   const ERL_NIF_TERM argv[])
{
   // validate parameters
   CRF_MODEL** resource = NULL;
   if (!enif_get_resource(env, argv[0], g_model_type, (void**)&resource))
      return enif_make_badarg(env);
   CRF_MODEL* model = *resource;
   ERL_NIF_TERM x = argv[1];
   unsigned n;
   CHECK(enif_get_list_length(env, x, &n), "invalid_x");
   // generate a model prediction from the source sequence
   crfsuite_dictionary_t* crf_attrs = NULL;
   crfsuite_dictionary_t* crf_labels = NULL;
   crfsuite_tagger_t* crf_tagger = NULL;
   crfsuite_instance_t crf_instance;
   int* path = NULL;
   ERL_NIF_TERM result;
   try {
      crfsuite_instance_init(&crf_instance);
      // create a new stateful tagger
      CHECKALLOC(model->crf->get_attrs(model->crf, &crf_attrs) == 0);
      CHECKALLOC(model->crf->get_labels(model->crf, &crf_labels) == 0);
      CHECKALLOC(model->crf->get_tagger(model->crf, &crf_tagger) == 0);
      // transfer the sequence to the tagger
      erl2crf_predict_instance(env, x, crf_attrs, &crf_instance);
      CHECKALLOC(crf_tagger->set(crf_tagger, &crf_instance) == 0);
      // predict the target sequence (path) and its score/lognorm
      path = nif_alloc<int>(n);
      double score;
      double lognorm;
      CHECK(crf_tagger->viterbi(crf_tagger, path, &score) == 0,
         "viterbi_failed");
      CHECK(crf_tagger->lognorm(crf_tagger, &lognorm) == 0,
         "lognorm_failed");
      // return the predicted sequence and its probability
      result = enif_make_tuple2(
         env,
         crf2erl_labels(env, crf_labels, path, n),
         enif_make_double(env, exp(score - lognorm)));
   } catch (NifError& e) {
      result = e.to_term(env);
   }
   // clean up
   if (crf_attrs != NULL)
      crf_attrs->release(crf_attrs);
   if (crf_labels != NULL)
      crf_labels->release(crf_labels);
   if (crf_tagger != NULL)
      crf_tagger->release(crf_tagger);
   crfsuite_instance_finish(&crf_instance);
   nif_free(path);
   return result;
}
/*-----------< FUNCTION: nif_destruct_model >--------------------------------
// Purpose:    frees the memory associated with a CRF model resource
// Parameters: env    - current erlang environment
//             object - model resource reference to free
// Returns:    none
---------------------------------------------------------------------------*/
void nif_destruct_model (ErlNifEnv* env, void* object)
{
   erl2crf_free_model(*(CRF_MODEL**)object);
}
/*-----------< FUNCTION: erl2crf_trainer >-----------------------------------
// Purpose:    creates a initialized CRF trainer
// Parameters: env     - current erlang environment
//             options - erlang CRF option map
// Returns:    allocated CRF trainer
---------------------------------------------------------------------------*/
crfsuite_trainer_t* erl2crf_trainer (
   ErlNifEnv*          env,
   const ERL_NIF_TERM& options)
{
   crfsuite_trainer_t *trainer = NULL;
   try {
      ERL_NIF_TERM key;
      ERL_NIF_TERM value;
      // retrieve the algorithm name
      key = enif_make_atom(env, "algorithm");
      CHECK(enif_get_map_value(env, options, key, &value),
         "missing_algorithm");
      // validate the algorithm name
      const char* algorithm = NULL;
      if (enif_is_identical(value, enif_make_atom(env, "lbfgs")))
         algorithm = "lbfgs";
      else if (enif_is_identical(value, enif_make_atom(env, "l2sgd")))
         algorithm = "l2sgd";
      else if (enif_is_identical(value, enif_make_atom(env, "ap")))
         algorithm = "averaged-perceptron";
      else if (enif_is_identical(value, enif_make_atom(env, "pa")))
         algorithm = "passive-aggressive";
      else if (enif_is_identical(value, enif_make_atom(env, "arow")))
         algorithm = "arow";
      else
         throw NifError("invalid_algorithm");
      // create the trainer instance
      char trainer_id[128 + 1];
      snprintf(
         trainer_id,
         sizeof(trainer_id) - 1,
         "train/crf1d/%s",
         algorithm);
      CHECKALLOC(crfsuite_create_instance(trainer_id, (void**)&trainer));
   } catch (NifError& e) {
      if (trainer != NULL)
         trainer->release(trainer);
      throw;
   }
   return trainer;
}
/*-----------< FUNCTION: erl2crf_params >------------------------------------
// Purpose:    populates a CRF parameters object
// Parameters: env     - current erlang environment
//             options - erlang CRF option map
//             trainer - crfsuite trainer to configure
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_params (
   ErlNifEnv*          env,
   const ERL_NIF_TERM& options,
   crfsuite_trainer_t* trainer)
{
   crfsuite_params_t* params = trainer->params(trainer);
   try {
      erl2crf_param_float(
         env,
         options,
         "min_freq",
         params,
         "feature.minfreq");
      erl2crf_param_bool(
         env,
         options,
         "all_possible_states?",
         params,
         "feature.possible_states");
      erl2crf_param_bool(
         env,
         options,
         "all_possible_transitions?",
         params,
         "feature.possible_transitions");
      erl2crf_param_float(
         env,
         options,
         "c1",
         params);
      erl2crf_param_float(
         env,
         options,
         "c2",
         params);
      erl2crf_param_int(
         env,
         options,
         "max_iterations",
         params);
      erl2crf_param_int(
         env,
         options,
         "num_memories",
         params);
      erl2crf_param_float(
         env,
         options,
         "epsilon",
         params);
      erl2crf_param_int(
         env,
         options,
         "period",
         params);
      erl2crf_param_float(
         env,
         options,
         "delta",
         params);
      erl2crf_param_string(
         env,
         options,
         "linesearch",
         params);
      erl2crf_param_int(
         env,
         options,
         "max_linesearch",
         params);
      erl2crf_param_float(
         env,
         options,
         "calibration_eta",
         params,
         "calibration.eta");
      erl2crf_param_float(
         env,
         options,
         "calibration_rate",
         params,
         "calibration.rate");
      erl2crf_param_int(
         env,
         options,
         "calibration_samples",
         params,
         "calibration.samples");
      erl2crf_param_int(
         env,
         options,
         "calibration_candidates",
         params,
         "calibration.candidates");
      erl2crf_param_int(
         env,
         options,
         "calibration_max_trials",
         params,
         "calibration.max_trials");
      erl2crf_param_int(
         env,
         options,
         "pa_type",
         params,
         "type");
      erl2crf_param_float(
         env,
         options,
         "c",
         params);
      erl2crf_param_bool(
         env,
         options,
         "error_sensitive?",
         params,
         "error_sensitive");
      erl2crf_param_bool(
         env,
         options,
         "averaging?",
         params,
         "averaging");
      erl2crf_param_float(
         env,
         options,
         "variance",
         params);
      erl2crf_param_float(
         env,
         options,
         "gamma",
         params);
      params->release(params);
   } catch (NifError& e) {
      params->release(params);
      throw;
   }
}
/*-----------< FUNCTION: erl2crf_param_bool >--------------------------------
// Purpose:    transfers an erlang boolean parameter to a CRF parameter
// Parameters: erl_env    - current erlang environment
//             erl_params - erlang parameter map
//             erl_name   - erlang parameter name
//             crf_params - crfsuite parameter object
//             crf_name   - crfsuite parameter name
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_param_bool(
   ErlNifEnv*          erl_env,
   const ERL_NIF_TERM& erl_params,
   const char*         erl_name,
   crfsuite_params_t*  crf_params,
   const char*         crf_name)
{
   ERL_NIF_TERM key = enif_make_atom(erl_env, erl_name);
   ERL_NIF_TERM value;
   CHECK(enif_get_map_value(erl_env, erl_params, key, &value), erl_name);
   crf_params->set_int(
      crf_params,
      crf_name ?: erl_name,
      enif_is_identical(value, enif_make_atom(erl_env, "true")) ? 1 : 0);
}
/*-----------< FUNCTION: erl2crf_param_int >---------------------------------
// Purpose:    transfers an erlang integer parameter to a CRF parameter
// Parameters: erl_env    - current erlang environment
//             erl_params - erlang parameter map
//             erl_name   - erlang parameter name
//             crf_params - crfsuite parameter object
//             crf_name   - crfsuite parameter name
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_param_int(
   ErlNifEnv*          erl_env,
   const ERL_NIF_TERM& erl_params,
   const char*         erl_name,
   crfsuite_params_t*  crf_params,
   const char*         crf_name)
{
   ERL_NIF_TERM key = enif_make_atom(erl_env, erl_name);
   ERL_NIF_TERM value;
   int n;
   CHECK(enif_get_map_value(erl_env, erl_params, key, &value), erl_name);
   CHECK(enif_get_int(erl_env, value, &n), erl_name);
   crf_params->set_int(crf_params, crf_name ?: erl_name, n);
}
/*-----------< FUNCTION: erl2crf_param_float >-------------------------------
// Purpose:    transfers an erlang double parameter to a CRF float parameter
// Parameters: erl_env    - current erlang environment
//             erl_params - erlang parameter map
//             erl_name   - erlang parameter name
//             crf_params - crfsuite parameter object
//             crf_name   - crfsuite parameter name
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_param_float(
   ErlNifEnv*          erl_env,
   const ERL_NIF_TERM& erl_params,
   const char*         erl_name,
   crfsuite_params_t*  crf_params,
   const char*         crf_name)
{
   ERL_NIF_TERM key = enif_make_atom(erl_env, erl_name);
   ERL_NIF_TERM value;
   double f;
   CHECK(enif_get_map_value(erl_env, erl_params, key, &value), erl_name);
   CHECK(enif_get_double(erl_env, value, &f), erl_name);
   crf_params->set_float(crf_params, crf_name ?: erl_name, f);
}
/*-----------< FUNCTION: erl2crf_param_string >------------------------------
// Purpose:    transfers an erlang atom parameter to a CRF string parameter
// Parameters: erl_env    - current erlang environment
//             erl_params - erlang parameter map
//             erl_name   - erlang parameter name
//             crf_params - crfsuite parameter object
//             crf_name   - crfsuite parameter name
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_param_string(
   ErlNifEnv*          erl_env,
   const ERL_NIF_TERM& erl_params,
   const char*         erl_name,
   crfsuite_params_t*  crf_params,
   const char*         crf_name)
{
   // get the atom from the map
   ERL_NIF_TERM key = enif_make_atom(erl_env, erl_name);
   ERL_NIF_TERM value;
   CHECK(enif_get_map_value(erl_env, erl_params, key, &value), erl_name);
   // get the atom string length
   unsigned cch = 0;
   CHECK(enif_get_atom_length(erl_env, value, &cch, ERL_NIF_LATIN1), erl_name);
   // fetch the string
   char sz[cch + 1];
   CHECK(enif_get_atom(erl_env, value, sz, cch + 1, ERL_NIF_LATIN1), erl_name);
   // copy it to crf parameters
   crf_params->set_string(crf_params, crf_name ?: erl_name, sz);
}
/*-----------< FUNCTION: erl2crf_train_data >--------------------------------
// Purpose:    transfers a list of training examples to the CRF structure
// Parameters: erl_env  - current erlang environment
//             x        - list of feature sequences
//             y        - list of label sequences
//             crf_data - CRF data structure to populate
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_train_data(
   ErlNifEnv*       erl_env,
   ERL_NIF_TERM     x,
   ERL_NIF_TERM     y,
   crfsuite_data_t* crf_data)
{
   crfsuite_data_init(crf_data);
   CHECKALLOC(crfsuite_create_instance(
      "dictionary",
      (void**)&crf_data->labels));
   CHECKALLOC(crfsuite_create_instance(
      "dictionary",
      (void**)&crf_data->attrs));
   // transfer each training example to the CRF data structure
   unsigned m;
   CHECK(enif_get_list_length(erl_env, x, &m), "invalid_x");
   for (int i = 0; i < m; i++) {
      ERL_NIF_TERM x_head, y_head;
      CHECK(enif_get_list_cell(erl_env, x, &x_head, &x), "invalid_x");
      CHECK(enif_get_list_cell(erl_env, y, &y_head, &y), "invalid_y");
      // copy the list heads to a new CRF instance
      crfsuite_instance_t crf_instance;
      try {
         crfsuite_instance_init(&crf_instance);
         erl2crf_train_instance(
            erl_env,
            x_head,
            y_head,
            crf_data->attrs,
            crf_data->labels,
            &crf_instance);
         CHECKALLOC(crfsuite_data_append(crf_data, &crf_instance) == 0);
         crfsuite_instance_finish(&crf_instance);
      } catch (NifError& e) {
         crfsuite_instance_finish(&crf_instance);
         throw;
      }
   }
}
/*-----------< FUNCTION: erl2crf_train_instance >----------------------------
// Purpose:    transfers a single training example to the CRF structure
// Parameters: erl_env      - current erlang environment
//             x_i          - list of features
//             y_i          - list of labels
//             crf_attrs    - CRF attribute dictionary
//             crf_labels   - CRF label dictionary
//             crf_instance - CRF instance structure to populate
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_train_instance(
   ErlNifEnv*             erl_env,
   ERL_NIF_TERM           x_i,
   ERL_NIF_TERM           y_i,
   crfsuite_dictionary_t* crf_attrs,
   crfsuite_dictionary_t* crf_labels,
   crfsuite_instance_t*   crf_instance)
{
   unsigned n;
   CHECK(enif_get_list_length(erl_env, x_i, &n), "invalid_x_i");
   crfsuite_instance_init_n(crf_instance, n);
   for (int i = 0; i < n; i++) {
      ERL_NIF_TERM x_i_head, y_i_head;
      CHECK(enif_get_list_cell(erl_env, x_i, &x_i_head, &x_i), "invalid_x_i");
      CHECK(enif_get_list_cell(erl_env, y_i, &y_i_head, &y_i), "invalid_y_i");
      erl2crf_features(erl_env, x_i_head, crf_attrs, crf_instance, i, true);
      erl2crf_label(erl_env, y_i_head, crf_labels, crf_instance, i);
   }
}
/*-----------< FUNCTION: erl2crf_predict_instance >--------------------------
// Purpose:    transfers a single inference example to the CRF structure
// Parameters: erl_env      - current erlang environment
//             x_i          - list of features
//             crf_attrs    - CRF attribute dictionary
//             crf_instance - CRF instance structure to populate
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_predict_instance(
   ErlNifEnv*             erl_env,
   ERL_NIF_TERM           x_i,
   crfsuite_dictionary_t* crf_attrs,
   crfsuite_instance_t*   crf_instance)
{
   unsigned n;
   CHECK(enif_get_list_length(erl_env, x_i, &n), "invalid_x_i");
   crfsuite_instance_init_n(crf_instance, n);
   for (int i = 0; i < n; i++) {
      ERL_NIF_TERM x_i_head;
      CHECK(enif_get_list_cell(erl_env, x_i, &x_i_head, &x_i), "invalid_x_i");
      erl2crf_features(erl_env, x_i_head, crf_attrs, crf_instance, i, false);
   }
}
/*-----------< FUNCTION: erl2crf_features >----------------------------------
// Purpose:    transfers a feature map to a CRF item structure
// Parameters: erl_env      - current erlang environment
//             erl_features - map of features
//             crf_attrs    - CRF attribute dictionary
//             crf_instance - CRF instance structure to populate
//             index        - sample sequence index
//             training     - training or predicting?
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_features(
   ErlNifEnv*             erl_env,
   const ERL_NIF_TERM&    erl_features,
   crfsuite_dictionary_t* crf_attrs,
   crfsuite_instance_t*   crf_instance,
   int                    index,
   bool                   training)
{
   crfsuite_item_t& crf_item = crf_instance->items[index];
   // create an iterator on the feature map
   size_t n;
   CHECK(enif_get_map_size(erl_env, erl_features, &n), "invalid_features");
   ErlNifMapIterator iterator;
   CHECK(enif_map_iterator_create(
         erl_env,
         erl_features,
         &iterator,
         ERL_NIF_MAP_ITERATOR_FIRST),
      "invalid_features");
   // transfer the feature map entries
   try {
      for (int i = 0; i < n; i++) {
         ERL_NIF_TERM k, v;
         CHECKALLOC(enif_map_iterator_get_pair(erl_env, &iterator, &k, &v));
         erl2crf_feature(erl_env, k, v, crf_attrs, crf_item, training);
         enif_map_iterator_next(erl_env, &iterator);
      }
      enif_map_iterator_destroy(erl_env, &iterator);
   } catch (NifError& e) {
      enif_map_iterator_destroy(erl_env, &iterator);
      throw;
   }
}
/*-----------< FUNCTION: erl2crf_feature >-----------------------------------
// Purpose:    transfers a feature instance to a CRF item structure
// Parameters: erl_env   - current erlang environment
//             erl_key   - feature key
//             erl_value - feature value
//             crf_attrs - CRF attribute dictionary
//             crf_item  - CRF item to populate
//             training  - training or predicting?
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_feature(
   ErlNifEnv*             erl_env,
   ERL_NIF_TERM&          erl_key,
   ERL_NIF_TERM&          erl_value,
   crfsuite_dictionary_t* crf_attrs,
   crfsuite_item_t&       crf_item,
   bool                   training)
{
   // decode the feature name
   ErlNifBinary feature_bin;
   CHECK(enif_inspect_binary(erl_env, erl_key, &feature_bin),
      "invalid_feature");
   char feature_name[feature_bin.size + 1];
   memcpy(feature_name, feature_bin.data, feature_bin.size);
   feature_name[feature_bin.size] = 0;
   // decode the feature value
   double feature_value;
   CHECK(enif_get_double(erl_env, erl_value, &feature_value),
      "invalid_feature");
   // if training, get/create a new feature id
   // otherwise, attempt to get an existing id and
   // ignore unseen feature names
   int aid = training
      ? crf_attrs->get(crf_attrs, feature_name)
      : crf_attrs->to_id(crf_attrs, feature_name);
   if (aid >= 0) {
      crfsuite_attribute_t attr;
      crfsuite_attribute_set(&attr, aid, feature_value);
      crfsuite_item_append_attribute(&crf_item, &attr);
   }
}
/*-----------< FUNCTION: erl2crf_label >-------------------------------------
// Purpose:    transfers a label value to a CRF instance
// Parameters: erl_env      - current erlang environment
//             ewrl_label   - label name (string)
//             crf_labels   - CRF label dictionary
//             crf_instance - CRF instance to populate
//             index        - sample sequence index
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_label(
   ErlNifEnv*             erl_env,
   const ERL_NIF_TERM&    erl_label,
   crfsuite_dictionary_t* crf_labels,
   crfsuite_instance_t*   crf_instance,
   int                    index)
{
   // decode the label name
   ErlNifBinary label_bin;
   CHECK(enif_inspect_binary(erl_env, erl_label, &label_bin), "invalid_label");
   char label_name[label_bin.size + 1];
   memcpy(label_name, label_bin.data, label_bin.size);
   label_name[label_bin.size] = 0;
   // register the label id with the instance
   crf_instance->labels[index] = crf_labels->get(crf_labels, label_name);
}
/*-----------< FUNCTION: erl2crf_free_train_data >---------------------------
// Purpose:    frees the memory associated with a CRF training data structure
// Parameters: data - CRF training data structure to free
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_free_train_data (crfsuite_data_t* data)
{
   if (data->attrs)
      data->attrs->release(data->attrs);
   if (data->labels)
      data->labels->release(data->labels);
   crfsuite_data_finish(data);
}
/*-----------< FUNCTION: erl2crf_free_model >--------------------------------
// Purpose:    frees the memory associated with a CRF model
// Parameters: model - CRF model structure to free
// Returns:    none
---------------------------------------------------------------------------*/
void erl2crf_free_model (CRF_MODEL* model)
{
   if (strlen(model->path) > 0)
      remove(model->path);
   if (model->crf)
      model->crf->release(model->crf);
   nif_free(model);
}
/*-----------< FUNCTION: crf_create_file >-----------------------------------
// Purpose:    generates a CRF model file name and opens it
// Parameters: path - return the model file path via here
// Returns:    an open file descriptor for the model file
---------------------------------------------------------------------------*/
int crf_create_file(char* path)
{
   strcpy(path, "/tmp/crf-XXXXXX");
   int fd = mkstemp(path);
   CHECK(fd != -1, "crf_create_file");
   return fd;
}
/*-----------< FUNCTION: crf2erl_labels >------------------------------------
// Purpose:    converts a CRF label sequence to a list of strings
// Parameters: erl_env    - current erlang environment
//             crf_labels - CRF label dictionary
//             crf_path   - list of CRF label identifiers for the sequence
//             n          - number of labels in the sequence
// Returns:    an erlang list of sequence label strings
---------------------------------------------------------------------------*/
ERL_NIF_TERM crf2erl_labels(
   ErlNifEnv*             erl_env,
   crfsuite_dictionary_t* crf_labels,
   int*                   crf_path,
   int                    n)
{
   ERL_NIF_TERM list = enif_make_list(erl_env, 0);
   for (int i = n - 1; i >= 0; i--) {
      // decode the sequence label identifier
      const char* label = NULL;
      CHECKALLOC(crf_labels->to_string(crf_labels, crf_path[i], &label) == 0);
      try {
         // create an erlang string for the label
         ErlNifBinary head;
         CHECKALLOC(enif_alloc_binary(strlen(label), &head));
         memcpy(head.data, label, head.size);
         // add the string to the list
         list = enif_make_list_cell(
            erl_env,
            enif_make_binary(erl_env, &head),
            list);
         crf_labels->free(crf_labels, label);
      } catch (NifError& e) {
         crf_labels->free(crf_labels, label);
         throw;
      }
   }
   return list;
}
