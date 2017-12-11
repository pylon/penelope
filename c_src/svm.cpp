/****************************************************************************
 *
 * MODULE:  svm.cpp
 * PURPOSE: nifs for libsvm
 *
 * for abbreviated names:
 * . m is the number of training examples
 * . n is the number of features
 * . k is the number of classes
 * . x is a feature matrix/vector/value
 * . y is a class vector/value
 *
 * see https://github.com/cjlin1/libsvm for details
 *
 ***************************************************************************/
/*-------------------[       Pre Include Defines       ]-------------------*/
/*-------------------[      Library Include Files      ]-------------------*/
/*-------------------[      Project Include Files      ]-------------------*/
#include "deps/libsvm/svm.h"
#include "penelope.hpp"
/*-------------------[      Macros/Constants/Types     ]-------------------*/
typedef struct svm_problem   SVM_PROBLEM;
typedef struct svm_model     SVM_MODEL;
typedef struct svm_node      SVM_NODE;
typedef struct svm_parameter SVM_PARAM;
/*-------------------[        Global Variables         ]-------------------*/
/*-------------------[        Global Prototypes        ]-------------------*/
/*-------------------[        Module Variables         ]-------------------*/
static ErlNifResourceType* g_model_type = NULL;
/*-------------------[        Module Prototypes        ]-------------------*/
static void erl2svm_problem (ErlNifEnv* env,
   ERL_NIF_TERM x,
   ERL_NIF_TERM y,
   SVM_PROBLEM* problem);
static SVM_MODEL* erl2svm_model (
   ErlNifEnv*   env,
   ERL_NIF_TERM params);
static void erl2svm_params (
   ErlNifEnv*   env,
   ERL_NIF_TERM options,
   SVM_PARAM*   params,
   int          training);
static SVM_NODE** erl2svm_features (
   ErlNifEnv*   env,
   ERL_NIF_TERM x,
   unsigned     m);
static SVM_NODE* erl2svm_feature (
   ErlNifEnv*   env,
   ERL_NIF_TERM x);
static double* erl2svm_targets (
   ErlNifEnv*   env,
   ERL_NIF_TERM y,
   unsigned     m);
static void erl2svm_free_problem (
   SVM_PROBLEM* problem);
static void erl2svm_free_params (
   SVM_PARAM* params);
static void erl2svm_free_model (
   SVM_MODEL* model);
static ERL_NIF_TERM svm2erl_model (
   ErlNifEnv* env,
   SVM_MODEL* model);
static SVM_MODEL* svm2svm_model (
   SVM_MODEL* source);
static void nif_destruct_model (
   ErlNifEnv* env,
   void*      object);
static void svm_print (
   const char* message);
/*-------------------[         Implementation          ]-------------------*/
/*-----------< FUNCTION: nif_svm_init >--------------------------------------
// Purpose:    svm module initialization
// Parameters: env - erlang environment
// Returns:    1 if successful
//             0 otherwise
---------------------------------------------------------------------------*/
int nif_svm_init (ErlNifEnv* env)
{
   // register the model resource type,
   // which holds trained SVM model instances
   ErlNifResourceFlags flags = ERL_NIF_RT_CREATE;
   g_model_type = enif_open_resource_type(
      env,
      NULL,
      "svm_model",
      &nif_destruct_model,
      flags,
      &flags);
   if (!g_model_type)
      return 0;
   // suppress libsvm debug output
   svm_set_print_string_function(&svm_print);
   return 1;
}
/*-----------< FUNCTION: nif_svm_train >-------------------------------------
// Purpose:    trains an SVM model
// Parameters: x      - list of feature vectors (floats)
//             y      - list of target labels (integer)
//             params - map of SVM parameters
// Returns:    reference to a trained SVM model resource
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_svm_train (
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
   // train the SVM model
   SVM_PROBLEM problem; memset(&problem, 0, sizeof(SVM_PROBLEM));
   SVM_PARAM   params;  memset(&params, 0, sizeof(SVM_PARAM));
   SVM_MODEL*  model    = NULL;
   SVM_MODEL** resource = NULL;
   ERL_NIF_TERM result;
   try {
      // extract training parameters and feature/target vectors
      erl2svm_problem(env, argv[0], argv[1], &problem);
      erl2svm_params(env, argv[2], &params, 1);
      const char* errors = svm_check_parameter(&problem, &params);
      if (errors)
         throw NifError(errors);
      // train the model
      model = svm_train(&problem, &params);
      // create an erlang resource to wrap the model
      CHECKALLOC(resource = (SVM_MODEL**)enif_alloc_resource(
         g_model_type,
         sizeof(SVM_MODEL*)));
      CHECKALLOC(*resource = svm2svm_model(model));
      result = enif_make_resource(env, resource);
      // relinquish the resource to erlang
      enif_release_resource(resource);
   } catch (NifError& e) {
      if (resource && *resource)
         erl2svm_free_model((SVM_MODEL*)*resource);
      result = e.to_term(env);
   }
   // free the model using the libsvm allocator
   if (model != NULL)
      svm_free_and_destroy_model(&model);
   // release the training parameters
   erl2svm_free_problem(&problem);
   erl2svm_free_params(&params);
   return result;
}
/*-----------< FUNCTION: nif_svm_export >------------------------------------
// Purpose:    extracts model parameters from an SVM resource,
//             which is useful for persisting a model externally
// Parameters: model - erlang resource wrapping the trained model
// Returns:    a map containing the model parameters
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_svm_export (
   ErlNifEnv*         env,
   int                argc,
   const ERL_NIF_TERM argv[])
{
   // validate parameters
   if (!enif_is_ref(env, argv[0]))
      return enif_make_badarg(env);
   // extract the model resource
   SVM_MODEL** resource = NULL;
   if (!enif_get_resource(env, argv[0], g_model_type, (void**)&resource))
      return enif_make_badarg(env);
   // convert the resource to a map
   try {
      return svm2erl_model(env, *resource);
   } catch (NifError& e) {
      return e.to_term(env);
   }
}
/*-----------< FUNCTION: nif_svm_compile >-----------------------------------
// Purpose:    converts the map representation of a model to the
//             native SVM model structure
// Parameters: model - map containing model parameters
// Returns:    reference to a trained SVM model resource
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_svm_compile (
   ErlNifEnv*         env,
   int                argc,
   const ERL_NIF_TERM argv[])
{
   // validate parameters
   if (!enif_is_map(env, argv[0]))
      return enif_make_badarg(env);
   // compile the model
   SVM_MODEL* model = NULL;
   try {
      model = erl2svm_model(env, argv[0]);
      // create an erlang resource to wrap the model
      SVM_MODEL** resource = (SVM_MODEL**)enif_alloc_resource(
         g_model_type,
         sizeof(SVM_MODEL*));
      CHECKALLOC(resource);
      *resource = model;
      ERL_NIF_TERM result = enif_make_resource(env, resource);
      // relinquish the resource to erlang
      enif_release_resource(resource);
      return result;
   } catch (NifError& e) {
      if (model)
         erl2svm_free_model(model);
      return e.to_term(env);
   }
}
/*-----------< FUNCTION: nif_svm_predict_class >-----------------------------
// Purpose:    predicts a single target class from a feature vector
// Parameters: model - reference to the trained SVM model
//             x     - feature vector to predict
// Returns:    predicted integer class
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_svm_predict_class (
   ErlNifEnv* env,
   int        argc,
   const ERL_NIF_TERM argv[])
{
   ERL_NIF_TERM result;
   SVM_NODE* features = NULL;
   SVM_MODEL** resource = NULL;
   // validate parameters
   if (!enif_get_resource(env, argv[0], g_model_type, (void**)&resource))
      return enif_make_badarg(env);
   SVM_MODEL* model = *resource;
   if (!enif_is_binary(env, argv[1]))
      return enif_make_badarg(env);
   try {
      // extract the feature vector
      features = erl2svm_feature(env, argv[1]);
      // predict the target class
      double cls = svm_predict(model, features);
      result = enif_make_int(env, (int)cls);
   } catch (NifError& e) {
      result = e.to_term(env);
   }
   nif_free(features);
   return result;
}
/*-----------< FUNCTION: nif_svm_predict_probability >-----------------------
// Purpose:    predicts class probabilities from a feature vector
// Parameters: model - reference to the trained SVM model
//             x     - feature vector to predict
// Returns:    list of probabilities (double) for each class, in the order
//             that the classes appear in the model
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_svm_predict_probability (
   ErlNifEnv* env,
   int        argc,
   const ERL_NIF_TERM argv[])
{
   ERL_NIF_TERM result;
   SVM_NODE* features = NULL;
   SVM_MODEL** resource = NULL;
   // validate parameters
   if (!enif_get_resource(env, argv[0], g_model_type, (void**)&resource))
      return enif_make_badarg(env);
   SVM_MODEL* model = *resource;
   if (!enif_is_binary(env, argv[1]))
      return enif_make_badarg(env);
   try {
      CHECK(svm_check_probability_model(model), "probability_not_trained");
      // extract the feature vector
      features = erl2svm_feature(env, argv[1]);
      // predict the class probabilities
      double prob[model->nr_class];
      svm_predict_probability(model, features, prob);
      // return the list of probabilities
      ERL_NIF_TERM results[model->nr_class];
      for (int i = 0; i < model->nr_class; i++)
         results[i] = enif_make_tuple2(env,
                        enif_make_int(env, model->label[i]),
                        enif_make_double(env, prob[i]));
      result = enif_make_list_from_array(env, results, model->nr_class);
   } catch (NifError& e) {
      result = e.to_term(env);
   }
   nif_free(features);
   return result;
}
/*-----------< FUNCTION: nif_destruct_model >--------------------------------
// Purpose:    frees the memory associated with an SVM model resource
// Parameters: env    - current erlang environment
//             object - model resource reference to free
// Returns:    none
---------------------------------------------------------------------------*/
void nif_destruct_model (ErlNifEnv* env, void* object)
{
   erl2svm_free_model(*(SVM_MODEL**)object);
}
/*-----------< FUNCTION: erl2svm_problem >-----------------------------------
// Purpose:    constrcts an SVM problem structure from feature/target vectors
// Parameters: env     - current erlang environment
//             x       - training feature vector list
//             y       - list of target class labels
//             problem - return the SVM problem via here
// Returns:    pointer to problem
---------------------------------------------------------------------------*/
void erl2svm_problem (
   ErlNifEnv*   env,
   ERL_NIF_TERM x,
   ERL_NIF_TERM y,
   SVM_PROBLEM* problem)
{
   unsigned m;
   CHECK(enif_get_list_length(env, x, &m), "invalid_x");
   problem->l = m;
   problem->x = erl2svm_features(env, x, m);
   problem->y = erl2svm_targets(env, y, m);
}
/*-----------< FUNCTION: erl2svm_model >-------------------------------------
// Purpose:    constrcts an SVM model structure from a map representation
// Parameters: env    - current erlang environment
//             params - model parameters (constructed via nif_svm_export)
// Returns:    pointer to the allocated and constructed model
---------------------------------------------------------------------------*/
SVM_MODEL* erl2svm_model (ErlNifEnv* env, ERL_NIF_TERM params)
{
   SVM_MODEL* model = nif_alloc<SVM_MODEL>();
   try {
      ERL_NIF_TERM key;
      ERL_NIF_TERM value;
      ERL_NIF_TERM tail;
      ErlNifBinary vector;
      // extract model parameters
      erl2svm_params(env, params, &model->param, 0);
      // extract model version
      int version = 0;
      key = enif_make_atom(env, "version");
      CHECK(enif_get_map_value(env, params, key, &value), "missing_version");
      CHECK(enif_get_int(env, value, &version), "invalid_version");
      CHECK(version == 1, "invalid_version");
      // extract class count
      unsigned class_count = 0;
      key = enif_make_atom(env, "classes");
      CHECK(enif_get_map_value(env, params, key, &tail), "missing_classes");
      CHECK(enif_get_list_length(env, tail, &class_count), "invalid_classes");
      model->nr_class = class_count;
      // extract class list
      model->label = nif_alloc<int>(model->nr_class);
      for (int i = 0; i < model->nr_class; i++) {
         CHECK(enif_get_list_cell(env, tail, &value, &tail), "missing_class");
         CHECK(enif_get_int(env, value, &model->label[i]), "invalid_class");
      }
      // extract support vector count
      key = enif_make_atom(env, "sv_count");
      CHECK(enif_get_map_value(env, params, key, &value), "missing_sv_count");
      CHECK(enif_get_int(env, value, &model->l), "invalid_sv_count");
      // extract class support vector counts
      key = enif_make_atom(env, "class_sv");
      CHECK(enif_get_map_value(env, params, key, &tail), "missing_label_svs");
      model->nSV = nif_alloc<int>(model->nr_class);
      for (int i = 0; i < model->nr_class; i++) {
         CHECK(enif_get_list_cell(env, tail, &value, &tail), "missing_label_sv");
         CHECK(enif_get_int(env, value, &model->nSV[i]), "invalid_label_sv");
      }
      // extract support vectors
      key = enif_make_atom(env, "sv");
      CHECK(enif_get_map_value(env, params, key, &tail), "missing_svs");
      model->SV = nif_alloc<SVM_NODE*>(model->l);
      for (int i = 0; i < model->l; i++) {
         CHECK(enif_get_list_cell(env, tail, &value, &tail), "missing_sv");
         CHECK(enif_inspect_binary(env, value, &vector), "invalid_sv");
         // copy the dense vector to the svm sparse list
         int n = vector.size / sizeof(float);
         model->SV[i] = nif_alloc<SVM_NODE>(n + 1);
         for (int j = 0; j < n; j++)
            model->SV[i][j] = (SVM_NODE){
               .index = j + 1,
               .value = ((float*)vector.data)[j]
            };
         model->SV[i][n] = (SVM_NODE){ .index = -1, .value = 0 };
      }
      // extract support vector coefficients
      int coef_count = model->nr_class - 1;
      key = enif_make_atom(env, "coef");
      CHECK(enif_get_map_value(env, params, key, &tail), "missing_coefs");
      model->sv_coef = nif_alloc<double*>(coef_count);
      for (int i = 0; i < coef_count; i++)
         model->sv_coef[i] = nif_alloc<double>(model->l);
      for (int i = 0; i < model->l; i++) {
         CHECK(enif_get_list_cell(env, tail, &value, &tail), "missing_coef");
         CHECK(enif_inspect_binary(env, value, &vector), "invalid_coef");
         // the coefficient matrix is transposed, so j before i
         for (int j = 0; j < model->nr_class - 1; j++)
            model->sv_coef[j][i] = ((float*)vector.data)[j];
      }
      // extract rho
      key = enif_make_atom(env, "rho");
      CHECK(enif_get_map_value(env, params, key, &value), "missing_rho");
      CHECK(enif_inspect_binary(env, value, &vector), "invalid_rho");
      model->rho = nif_alloc<double>(vector.size / sizeof(float));
      for (int i = 0; i < (int)(vector.size / sizeof(float)); i++)
         model->rho[i] = ((float*)vector.data)[i];
      // extract probA
      key = enif_make_atom(env, "prob_a");
      CHECK(enif_get_map_value(env, params, key, &value), "missing_prob_a");
      if (!enif_is_identical(value, enif_make_atom(env, "nil"))) {
         CHECK(enif_inspect_binary(env, value, &vector), "invalid_prob_a");
         model->probA = nif_alloc<double>(vector.size / sizeof(float));
         for (int i = 0; i < (int)(vector.size / sizeof(float)); i++)
            model->probA[i] = ((float*)vector.data)[i];
      }
      // extract probB
      key = enif_make_atom(env, "prob_b");
      CHECK(enif_get_map_value(env, params, key, &value), "missing_prob_b");
      if (!enif_is_identical(value, enif_make_atom(env, "nil"))) {
         CHECK(enif_inspect_binary(env, value, &vector), "invalid_prob_b");
         model->probB = nif_alloc<double>(vector.size / sizeof(float));
         for (int i = 0; i < (int)(vector.size / sizeof(float)); i++)
            model->probB[i] = ((float*)vector.data)[i];
      }
      return model;
   } catch (NifError& e) {
      erl2svm_free_model(model);
      throw;
   }
}
/*-----------< FUNCTION: erl2svm_params >------------------------------------
// Purpose:    constrcts an SVM param structure an options map
// Parameters: env      - current erlang environment
//             options  - SVM options map
//             params   - returns SVM parameters via here
//             training - true if the model is being trained
//                        false if we are loading it from a trained model
// Returns:    pointer to params
---------------------------------------------------------------------------*/
void erl2svm_params (
   ErlNifEnv*   env,
   ERL_NIF_TERM options,
   SVM_PARAM*   params,
   int          training) {
   ERL_NIF_TERM key;
   ERL_NIF_TERM value;
   params->svm_type = C_SVC;
   // decode kernel type
   key = enif_make_atom(env, "kernel");
   CHECK(enif_get_map_value(env, options, key, &value), "missing_kernel");
   if (enif_is_identical(value, enif_make_atom(env, "linear")))
      params->kernel_type = LINEAR;
   else if (enif_is_identical(value, enif_make_atom(env, "poly")))
      params->kernel_type = POLY;
   else if (enif_is_identical(value, enif_make_atom(env, "rbf")))
      params->kernel_type = RBF;
   else if (enif_is_identical(value, enif_make_atom(env, "sigmoid")))
      params->kernel_type = SIGMOID;
   else
      throw NifError("invalid_kernel");
   // decode kernel parameters
   key = enif_make_atom(env, "degree");
   CHECK(enif_get_map_value(env, options, key, &value), "missing_degree");
   CHECK(enif_get_int(env, value, &params->degree), "invalid_degree");
   key = enif_make_atom(env, "gamma");
   CHECK(enif_get_map_value(env, options, key, &value), "missing_gamma");
   CHECK(enif_get_double(env, value, &params->gamma), "invalid_gamma");
   key = enif_make_atom(env, "coef0");
   CHECK(enif_get_map_value(env, options, key, &value), "missing_coef0");
   CHECK(enif_get_double(env, value, &params->coef0), "invalid_coef0");
   // decode training parameters
   if (training) {
      // decode cost parameter
      key = enif_make_atom(env, "c");
      CHECK(enif_get_map_value(env, options, key, &value), "missing_c");
      CHECK(enif_get_double(env, value, &params->C), "invalid_c");
      // decode class weights
      size_t weight_count;
      key = enif_make_atom(env, "weights");
      CHECK(enif_get_map_value(env, options, key, &value), "missing_weights");
      CHECK(enif_get_map_size(env, value, &weight_count), "invalid_weights");
      if (weight_count > 0) {
         params->nr_weight    = weight_count;
         params->weight_label = nif_alloc<int>(weight_count);
         params->weight       = nif_alloc<double>(weight_count);
         ErlNifMapIterator iter;
         CHECKALLOC(enif_map_iterator_create(
            env, value, &iter, ERL_NIF_MAP_ITERATOR_FIRST));
         try {
            for (int i = 0; i < (int)weight_count; i++) {
               CHECK(enif_map_iterator_get_pair(
                  env, &iter, &key, &value),
                  "invalid_weight");
               CHECK(enif_get_int(
                  env, key, &params->weight_label[i]),
                  "invalid_weight");
               CHECK(enif_get_double(
                  env, value, &params->weight[i]),
                  "invalid_weight");
               enif_map_iterator_next(env, &iter);
            }
            enif_map_iterator_destroy(env, &iter);
         } catch (...) {
            enif_map_iterator_destroy(env, &iter);
            throw;
         }
      }
      // decode training parameters
      key = enif_make_atom(env, "epsilon");
      CHECK(enif_get_map_value(env, options, key, &value), "missing_epsilon");
      CHECK(enif_get_double(env, value, &params->eps));
      key = enif_make_atom(env, "cache_size");
      CHECK(enif_get_map_value(env, options, key, &value), "missing_cache");
      CHECK(enif_get_double(env, value, &params->cache_size), "invalid_cache");
      key = enif_make_atom(env, "shrinking?");
      CHECK(enif_get_map_value(env, options, key, &value), "missing_shrink");
      CHECK(enif_is_atom(env, value), "invalid_shrink");
      if (enif_is_identical(value, enif_make_atom(env, "true")))
         params->shrinking = 1;
      key = enif_make_atom(env, "probability?");
      CHECK(enif_get_map_value(env, options, key, &value), "missing_prob");
      CHECK(enif_is_atom(env, value), "invalid_prob");
      if (enif_is_identical(value, enif_make_atom(env, "true")))
         params->probability = 1;
   }
}
/*-----------< FUNCTION: erl2svm_features >----------------------------------
// Purpose:    converts a list of feature vectors to an SVM sparse matrix
// Parameters: env - current erlang environment
//             x   - list of feature vectors (floats)
//             m   - number of vectors in the feature matrix
// Returns:    pointer to an array of sparse feature vectors
---------------------------------------------------------------------------*/
SVM_NODE** erl2svm_features (ErlNifEnv* env, ERL_NIF_TERM x, unsigned m) {
   SVM_NODE** nodes = nif_alloc<SVM_NODE*>(m);
   try {
      for (int i = 0; i < (int)m; i++) {
         ERL_NIF_TERM head;
         CHECK(enif_get_list_cell(env, x, &head, &x), "missing_features");
         nodes[i] = erl2svm_feature(env, head);
      }
      return nodes;
   } catch (...) {
      for (int i = 0; i < (int)m; i++)
         nif_free(nodes[i]);
      nif_free(nodes);
      throw;
   }
}
/*-----------< FUNCTION: erl2svm_feature >-----------------------------------
// Purpose:    converts a feature vectors to an SVM sparse vector
// Parameters: env - current erlang environment
//             x   - feature vector (floats)
// Returns:    pointer to a sparse feature vector
---------------------------------------------------------------------------*/
SVM_NODE* erl2svm_feature (ErlNifEnv* env, ERL_NIF_TERM x) {
   ErlNifBinary vector;
   CHECK(enif_inspect_binary(env, x, &vector), "invalid_feature");
   // copy the feature vector to the sparse array
   int n = vector.size / sizeof(float);
   SVM_NODE* nodes = nif_alloc<SVM_NODE>(n + 1);
   for (int j = 0; j < n; j++)
      nodes[j] = (SVM_NODE){
         .index = j + 1,
         .value = ((float*)vector.data)[j]
      };
   // terminate the sparse vector with -1 per libsvm spec
   nodes[n] = (SVM_NODE){ .index = -1, .value = 0 };
   return nodes;
}
/*-----------< FUNCTION: erl2svm_targets >-----------------------------------
// Purpose:    converts a list of target labels to an array of SVM labels
// Parameters: env - current erlang environment
//             y   - list of target labels (integers)
//             m   - number of training labels
// Returns:    array of doubles representing the target labels
---------------------------------------------------------------------------*/
double* erl2svm_targets (ErlNifEnv* env, ERL_NIF_TERM y, unsigned m) {
   double* targets = nif_alloc<double>(m);
   try {
      for (int i = 0; i < (int)m; i++) {
         // get the list head and advance the tail
         ERL_NIF_TERM head;
         CHECK(enif_get_list_cell(env, y, &head, &y), "missing_target");
         // retrieve the target label value
         int cls;
         CHECK(enif_get_int(env, head, &cls), "invalid_target");
         targets[i] = cls;
      }
      return targets;
   } catch (...) {
      nif_free(targets);
      throw;
   }
}
/*-----------< FUNCTION: erl2svm_free_problem >------------------------------
// Purpose:    frees the memory associated with an SVM problem structure
// Parameters: problem - structure to free
// Returns:    none
---------------------------------------------------------------------------*/
void erl2svm_free_problem (SVM_PROBLEM* problem)
{
   if (problem->x)
      for (int i = 0; i < problem->l; i++)
         nif_free(problem->x[i]);
   nif_free(problem->x);
   problem->x = NULL;
   nif_free(problem->y);
   problem->y = NULL;
}
/*-----------< FUNCTION: erl2svm_free_params >-------------------------------
// Purpose:    frees the memory associated with an SVM parameter structure
// Parameters: params - structure to free
// Returns:    none
---------------------------------------------------------------------------*/
void erl2svm_free_params (SVM_PARAM* params)
{
   nif_free(params->weight_label);
   params->weight_label = NULL;
   nif_free(params->weight);
   params->weight = NULL;
}
/*-----------< FUNCTION: svm2erl_model >-------------------------------------
// Purpose:    converts an SVM model to an erlang map
// Parameters: env   - current erlang environment
//             model - SVM model structure to convert
// Returns:    erlang map containing the model parameters
---------------------------------------------------------------------------*/
ERL_NIF_TERM svm2erl_model (ErlNifEnv* env, SVM_MODEL* model)
{
   ERL_NIF_TERM result = enif_make_new_map(env);
   ERL_NIF_TERM key;
   ERL_NIF_TERM value;
   ErlNifBinary vector;
   // encode version
   key   = enif_make_atom(env, "version");
   value = enif_make_int(env, 1);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode kernel
   key = enif_make_atom(env, "kernel");
   switch (model->param.kernel_type) {
      case POLY:    value = enif_make_atom(env, "poly"); break;
      case RBF:     value = enif_make_atom(env, "rbf"); break;
      case SIGMOID: value = enif_make_atom(env, "sigmoid"); break;
      default:      value = enif_make_atom(env, "linear"); break;
   }
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode degree
   key   = enif_make_atom(env, "degree");
   value = enif_make_int(env, model->param.degree);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode gamma
   key   = enif_make_atom(env, "gamma");
   value = enif_make_double(env, model->param.gamma);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode coef0
   key   = enif_make_atom(env, "coef0");
   value = enif_make_double(env, model->param.coef0);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode classes
   ERL_NIF_TERM classes[model->nr_class]; memset(classes, 0, sizeof(classes));
   for (int i = 0; i < model->nr_class; i++)
      classes[i] = enif_make_int(env, model->label[i]);
   key   = enif_make_atom(env, "classes");
   value = enif_make_list_from_array(env, classes, model->nr_class);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode vector count
   key   = enif_make_atom(env, "sv_count");
   value = enif_make_int(env, model->l);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode class support vector counts
   ERL_NIF_TERM label_sv[model->nr_class]; memset(label_sv, 0, sizeof(label_sv));
   for (int i = 0; i < model->nr_class; i++)
      label_sv[i] = enif_make_int(env, model->nSV[i]);
   key   = enif_make_atom(env, "class_sv");
   value = enif_make_list_from_array(env, label_sv, model->nr_class);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode support vectors
   ERL_NIF_TERM vectors[model->l]; memset(vectors, 0, sizeof(vectors));
   for (int i = 0; i < model->l; i++) {
      int n = 0;
      for (SVM_NODE* node = model->SV[0]; node->index != -1; node++)
         n++;
      CHECKALLOC(enif_alloc_binary(n * sizeof(float), &vector));
      for (int j = 0; j < n; j++)
         ((float*)vector.data)[j] = model->SV[i][j].value;
      vectors[i] = enif_make_binary(env, &vector);
   }
   key   = enif_make_atom(env, "sv");
   value = enif_make_list_from_array(env, vectors, model->l);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode support vector coefficients
   ERL_NIF_TERM coefs[model->l]; memset(coefs, 0, sizeof(coefs));
   for (int i = 0; i < model->l; i++) {
      int coef_count = model->nr_class - 1;
      CHECKALLOC(enif_alloc_binary(coef_count * sizeof(float), &vector));
      // the coefficient matrix is transposed, so j before i
      for (int j = 0; j < coef_count; j++)
         ((float*)vector.data)[j] = model->sv_coef[j][i];
      coefs[i] = enif_make_binary(env, &vector);
   }
   key   = enif_make_atom(env, "coef");
   value = enif_make_list_from_array(env, coefs, model->l);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode rho
   int rho_count = model->nr_class * (model->nr_class - 1) / 2;
   CHECKALLOC(enif_alloc_binary(rho_count * sizeof(float), &vector));
   for (int i = 0; i < rho_count; i++)
      ((float*)vector.data)[i] = model->rho[i];
   key   = enif_make_atom(env, "rho");
   value = enif_make_binary(env, &vector);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode prob_a
   key = enif_make_atom(env, "prob_a");
   if (model->probA) {
      int prob_count = model->nr_class * (model->nr_class - 1) / 2;
      CHECKALLOC(enif_alloc_binary(prob_count * sizeof(float), &vector));
      for (int i = 0; i < prob_count; i++)
         ((float*)vector.data)[i] = model->probA[i];
      value = enif_make_binary(env, &vector);
      CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   } else {
      value = enif_make_atom(env, "nil");
      CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   }
   // encode prob_b
   key = enif_make_atom(env, "prob_b");
   if (model->probB) {
      int prob_count = model->nr_class * (model->nr_class - 1) / 2;
      CHECKALLOC(enif_alloc_binary(prob_count * sizeof(float), &vector));
      for (int i = 0; i < prob_count; i++)
         ((float*)vector.data)[i] = model->probB[i];
      value = enif_make_binary(env, &vector);
      CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   } else {
      value = enif_make_atom(env, "nil");
      CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   }
   return result;
}
/*-----------< FUNCTION: svm2svm_model >-------------------------------------
// Purpose:    clones an SVM  model structure
//             this is needed because svm_train borrows vectors
//             from the training matrix so that it can't be freed
// Parameters: source - SVM model structure to copy
// Returns:    cloned SVM model
---------------------------------------------------------------------------*/
SVM_MODEL* svm2svm_model (SVM_MODEL* source)
{
   SVM_MODEL* target = nif_alloc<SVM_MODEL>();
   try {
      // copy scalar fields + clear class weights (training-only)
      memcpy(&target->param, &source->param, sizeof(SVM_PARAM));
      target->nr_class           = source->nr_class;
      target->l                  = source->l;
      target->param.nr_weight    = 0;
      target->param.weight_label = NULL;
      target->param.weight       = NULL;
      // copy support vectors
      target->SV = nif_alloc<SVM_NODE*>(target->l);
      for (int i = 0; i < target->l; i++) {
         int n = 0;
         for (SVM_NODE* node = source->SV[0]; node->index != -1; node++)
            n++;
         target->SV[i] = nif_alloc<SVM_NODE>(n + 1);
         memcpy(target->SV[i], source->SV[i], (n + 1) * sizeof(SVM_NODE));
      }
      // copy coefficients
      int coef_count = target->nr_class - 1;
      target->sv_coef = nif_alloc<double*>(coef_count);
      for (int i = 0; i < coef_count; i++)
         target->sv_coef[i] = nif_clone(source->sv_coef[i], target->l);
      // copy rho/probabilities
      int pair_count = target->nr_class * (target->nr_class - 1) / 2;
      target->rho = nif_clone(source->rho, pair_count);
      if (source->probA) {
         target->probA = nif_clone(source->probA, pair_count);
         target->probB = nif_clone(source->probB, pair_count);
      }
      // copy labels/vector count
      target->label = nif_clone(source->label, target->nr_class);
      target->nSV   = nif_clone(source->nSV, target->nr_class);
   } catch (...) {
      erl2svm_free_model(target);
      throw;
   }
   return target;
}
/*-----------< FUNCTION: erl2svm_free_model >--------------------------------
// Purpose:    frees the memory associated with an SVM  model
// Parameters: model - SVM model structure to free
// Returns:    none
---------------------------------------------------------------------------*/
void erl2svm_free_model (SVM_MODEL* model)
{
   erl2svm_free_params(&model->param);
   if (model->SV)
      for (int i = 0; model->SV && i < model->l; i++)
         nif_free(model->SV[i]);
   nif_free(model->SV);
   if (model->sv_coef)
      for (int i = 0; i < model->nr_class - 1; i++)
         nif_free(model->sv_coef[i]);
   nif_free(model->sv_coef);
   nif_free(model->rho);
   nif_free(model->probA);
   nif_free(model->probB);
   nif_free(model->label);
   nif_free(model->nSV);
   nif_free(model);
}
/*-----------< FUNCTION: svm_print >-----------------------------------------
// Purpose:    libsvm debug output callback
// Parameters: message - message to display
// Returns:    none
---------------------------------------------------------------------------*/
void svm_print (const char* message) {
   // suppress debug output
}
