/****************************************************************************
 *
 * MODULE:  lin.cpp
 * PURPOSE: nifs for liblinear
 *
 * for abbreviated names:
 * . m is the number of training examples
 * . n is the number of features
 * . k is the number of classes
 * . x is a feature matrix/vector/value
 * . y is a class vector/value
 *
 * The basic functionality of liblinear is extended to include predicting
 * calibrated probabilites for SVM models, using Platt scaling. Binary Platt
 * scaling is extended to OVR multiclass using simple normalization.
 *
 * see https://github.com/cjlin1/liblinear for details
 *
 ***************************************************************************/
/*-------------------[       Pre Include Defines       ]-------------------*/
/*-------------------[      Library Include Files      ]-------------------*/
#include <math.h>
/*-------------------[      Project Include Files      ]-------------------*/
#include "deps/liblinear/linear.h"
#include "penelope.hpp"
/*-------------------[      Macros/Constants/Types     ]-------------------*/
// extend the linear model structure to include an optional calibration model
typedef struct tag_model : model {
   double* prob_a;
   double* prob_b;
} LINEAR_MODEL;
typedef struct problem      LINEAR_PROBLEM;
typedef struct feature_node LINEAR_NODE;
typedef struct parameter    LINEAR_PARAM;
/*-------------------[        Global Variables         ]-------------------*/
/*-------------------[        Global Prototypes        ]-------------------*/
/*-------------------[        Module Variables         ]-------------------*/
static ErlNifResourceType* g_model_type = NULL;
/*-------------------[        Module Prototypes        ]-------------------*/
static bool erl2lin_must_calibrate (
   ErlNifEnv*    env,
   ERL_NIF_TERM  options,
   LINEAR_PARAM& params);
static void erl2lin_problem (
   ErlNifEnv*      env,
   ERL_NIF_TERM    x,
   ERL_NIF_TERM    y,
   ERL_NIF_TERM    params,
   LINEAR_PROBLEM* problem);
static LINEAR_MODEL* erl2lin_model (
   ErlNifEnv*   env,
   ERL_NIF_TERM params);
static void erl2lin_params (
   ErlNifEnv*    env,
   ERL_NIF_TERM  options,
   LINEAR_PARAM* params,
   int           training);
static LINEAR_NODE** erl2lin_features (
   ErlNifEnv*   env,
   ERL_NIF_TERM x,
   unsigned     m,
   double       bias);
static LINEAR_NODE* erl2lin_feature (
   ErlNifEnv*   env,
   ERL_NIF_TERM x,
   double       bias);
static double* erl2lin_targets (
   ErlNifEnv*   env,
   ERL_NIF_TERM y,
   unsigned     m);
static void erl2lin_free_problem (
   LINEAR_PROBLEM* problem);
static void erl2lin_free_params (
   LINEAR_PARAM* params);
static void erl2lin_free_model (
   LINEAR_MODEL* model);
static ERL_NIF_TERM lin2erl_model (
   ErlNifEnv*    env,
   LINEAR_MODEL* model);
static LINEAR_MODEL* lin2lin_model (
   model*  source,
   double* prob_a,
   double* prob_b);
static void nif_destruct_model (
   ErlNifEnv* env,
   void*      object);
static void lin_print (
   const char* message);
static void lin_calibrate_train (
   int     m,
   double* decision,
   double* labels,
   double& prob_a,
   double& prob_b);
static double lin_calibrate_predict (
   double decision,
   double prob_a,
   double prob_b);
/*-------------------[         Implementation          ]-------------------*/
/*-----------< FUNCTION: nif_lin_init >--------------------------------------
// Purpose:    linear module initialization
// Parameters: env - erlang environment
// Returns:    1 if successful
//             0 otherwise
---------------------------------------------------------------------------*/
int nif_lin_init (ErlNifEnv* env)
{
   // register the model resource type,
   // which holds trained linear model instances
   ErlNifResourceFlags flags = ERL_NIF_RT_CREATE;
   g_model_type = enif_open_resource_type(
      env,
      NULL,
      "lin_model",
      &nif_destruct_model,
      flags,
      &flags);
   if (!g_model_type)
      return 0;
   // suppress liblinear debug output
   set_print_string_function(&lin_print);
   return 1;
}
/*-----------< FUNCTION: nif_lin_train >-------------------------------------
// Purpose:    trains a linear model
// Parameters: x      - list of feature vectors (floats)
//             y      - list of target labels (integer)
//             params - map of linear parameters
// Returns:    reference to a trained linear model resource
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_lin_train (
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
   // train the linear model
   LINEAR_PROBLEM problem; memset(&problem, 0, sizeof(LINEAR_PROBLEM));
   LINEAR_PARAM   params;  memset(&params, 0, sizeof(LINEAR_PARAM));
   model*         linear   = NULL;
   double*        prob_a   = NULL;
   double*        prob_b   = NULL;
   double*        prob_d   = NULL;
   double*        prob_l   = NULL;
   LINEAR_MODEL** resource = NULL;
   ERL_NIF_TERM result;
   try {
      // extract training parameters and feature/target vectors
      erl2lin_problem(env, argv[0], argv[1], argv[2], &problem);
      erl2lin_params(env, argv[2], &params, 1);
      const char* errors = check_parameter(&problem, &params);
      if (errors)
         throw NifError(errors);
      // train the prediction model
      linear = CHECKALLOC(train(&problem, &params));
      // train the calibration model
      if (erl2lin_must_calibrate(env, argv[2], params)) {
         int model_count = linear->nr_class == 2 ? 1 : linear->nr_class;
         prob_a = nif_alloc<double>(model_count);
         prob_b = nif_alloc<double>(model_count);
         prob_d = nif_alloc<double>(problem.l);
         prob_l = nif_alloc<double>(problem.l);
         for (int i = 0; i < model_count; i++) {
            // train a calibration logistic regression model per class
            for (int j = 0; j < problem.l; j++) {
               double decision[model_count];
               double label = predict_values(linear, problem.x[j], decision);
               prob_d[j] = decision[i];
               prob_l[j] = label == linear->label[i] ? 1 : -1;
            }
            lin_calibrate_train(
               problem.l,
               prob_d,
               prob_l,
               prob_a[i],
               prob_b[i]);
         }
      }
      // create an erlang resource to wrap the model
      CHECKALLOC(resource = (LINEAR_MODEL**)enif_alloc_resource(
         g_model_type,
         sizeof(LINEAR_MODEL*)));
      CHECKALLOC(*resource = lin2lin_model(linear, prob_a, prob_b));
      prob_a = NULL;
      prob_b = NULL;
      result = enif_make_resource(env, resource);
      // relinquish the resource to erlang
      enif_release_resource(resource);
   } catch (NifError& e) {
      if (resource && *resource)
         erl2lin_free_model((LINEAR_MODEL*)*resource);
      result = e.to_term(env);
   }
   // free the model using the liblinear allocator
   if (linear != NULL)
      free_and_destroy_model(&linear);
   if (prob_a != NULL) {
      nif_free(prob_a);
      nif_free(prob_b);
   }
   nif_free(prob_d);
   nif_free(prob_l);
   // release the training parameters
   erl2lin_free_problem(&problem);
   erl2lin_free_params(&params);
   return result;
}
/*-----------< FUNCTION: nif_lin_export >------------------------------------
// Purpose:    extracts model parameters from a linear model resource,
//             which is useful for persisting a model externally
// Parameters: model - erlang resource wrapping the trained model
// Returns:    a map containing the model parameters
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_lin_export (
   ErlNifEnv*         env,
   int                argc,
   const ERL_NIF_TERM argv[])
{
   // validate parameters
   if (!enif_is_ref(env, argv[0]))
      return enif_make_badarg(env);
   // extract the model resource
   LINEAR_MODEL** resource = NULL;
   if (!enif_get_resource(env, argv[0], g_model_type, (void**)&resource))
      return enif_make_badarg(env);
   // convert the resource to a map
   try {
      return lin2erl_model(env, *resource);
   } catch (NifError& e) {
      return e.to_term(env);
   }
}
/*-----------< FUNCTION: nif_lin_compile >-----------------------------------
// Purpose:    converts the map representation of a model to the
//             native linear model structure
// Parameters: model - map containing model parameters
// Returns:    reference to a trained linear model resource
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_lin_compile (
   ErlNifEnv*         env,
   int                argc,
   const ERL_NIF_TERM argv[])
{
   // validate parameters
   if (!enif_is_map(env, argv[0]))
      return enif_make_badarg(env);
   // compile the model
   LINEAR_MODEL* model = NULL;
   try {
      model = erl2lin_model(env, argv[0]);
      // create an erlang resource to wrap the model
      LINEAR_MODEL** resource = (LINEAR_MODEL**)enif_alloc_resource(
         g_model_type,
         sizeof(LINEAR_MODEL*));
      CHECKALLOC(resource);
      *resource = model;
      ERL_NIF_TERM result = enif_make_resource(env, resource);
      // relinquish the resource to erlang
      enif_release_resource(resource);
      return result;
   } catch (NifError& e) {
      if (model)
         erl2lin_free_model(model);
      return e.to_term(env);
   }
}
/*-----------< FUNCTION: nif_lin_predict_class >-----------------------------
// Purpose:    predicts a single target class from a feature vector
// Parameters: model - reference to the trained linear model
//             x     - feature vector to predict
// Returns:    predicted integer class
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_lin_predict_class (
   ErlNifEnv* env,
   int        argc,
   const ERL_NIF_TERM argv[])
{
   ERL_NIF_TERM result;
   LINEAR_NODE* features = NULL;
   LINEAR_MODEL** resource = NULL;
   // validate parameters
   if (!enif_get_resource(env, argv[0], g_model_type, (void**)&resource))
      return enif_make_badarg(env);
   LINEAR_MODEL* model = *resource;
   if (!enif_is_binary(env, argv[1]))
      return enif_make_badarg(env);
   try {
      // extract the feature vector
      features = erl2lin_feature(env, argv[1], model->bias);
      // predict the target class
      double cls = predict(model, features);
      result = enif_make_int(env, (int)cls);
   } catch (NifError& e) {
      result = e.to_term(env);
   }
   nif_free(features);
   return result;
}
/*-----------< FUNCTION: nif_lin_predict_probability >-----------------------
// Purpose:    predicts class probabilities from a feature vector
// Parameters: model - reference to the trained linear model
//             x     - feature vector to predict
// Returns:    list of probabilities (double) for each class, in the order
//             that the classes appear in the model
---------------------------------------------------------------------------*/
ERL_NIF_TERM nif_lin_predict_probability (
   ErlNifEnv* env,
   int        argc,
   const ERL_NIF_TERM argv[])
{
   ERL_NIF_TERM result;
   LINEAR_NODE* features = NULL;
   LINEAR_MODEL** resource = NULL;
   // validate parameters
   if (!enif_get_resource(env, argv[0], g_model_type, (void**)&resource))
      return enif_make_badarg(env);
   LINEAR_MODEL* model = *resource;
   if (!enif_is_binary(env, argv[1]))
      return enif_make_badarg(env);
   try {
      CHECK(check_probability_model(model) || model->prob_a,
         "probability_not_trained");
      // extract the feature vector
      features = erl2lin_feature(env, argv[1], model->bias);
      // predict the class probabilities directly if supported
      double prob[model->nr_class];
      if (check_probability_model(model))
         predict_probability(model, features, prob);
      else {
         // otherwise, compute calibrated probabilities
         int model_count = model->nr_class == 2 ? 1 : model->nr_class;
         double decision[model_count];
         predict_values(model, features, decision);
         for (int i = 0; i < model_count; i++)
            prob[i] = lin_calibrate_predict(
               decision[i],
               model->prob_a[i],
               model->prob_b[i]);
         // normalize the calibrated probabilities
         if (model_count == 1)
            prob[1] = 1 - prob[0];
         else {
            double sum = 0;
            for (int i = 0; i < model->nr_class; i++)
               sum += prob[i];
            for (int i = 0; i < model->nr_class; i++)
               prob[i] = prob[i] / sum;
         }
      }
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
// Purpose:    frees the memory associated with a linear model resource
// Parameters: env    - current erlang environment
//             object - model resource reference to free
// Returns:    none
---------------------------------------------------------------------------*/
void nif_destruct_model (ErlNifEnv* env, void* object)
{
   erl2lin_free_model(*(LINEAR_MODEL**)object);
}
/*-----------< FUNCTION: erl2lin_must_calibrate >----------------------------
// Purpose:    determines whether a calibration model should be built
// Parameters: env     - current erlang environment
//             options - training parameters
//             params  - linear model parameters
// Returns:    true if a calibration model is required
//             false otherwise
---------------------------------------------------------------------------*/
bool erl2lin_must_calibrate (
   ErlNifEnv*    env,
   ERL_NIF_TERM  options,
   LINEAR_PARAM& params)
{
   ERL_NIF_TERM key;
   ERL_NIF_TERM value;
   switch (params.solver_type) {
      case L2R_L2LOSS_SVC_DUAL:
      case L2R_L2LOSS_SVC:
      case L2R_L1LOSS_SVC_DUAL:
      case MCSVM_CS:
      case L1R_L2LOSS_SVC:
         key = enif_make_atom(env, "probability?");
         CHECK(enif_get_map_value(env, options, key, &value), "missing_prob");
         CHECK(enif_is_atom(env, value), "invalid_prob");
         return enif_is_identical(value, enif_make_atom(env, "true"))
            ? true
            : false;
      default:
         return false;
   }
}
/*-----------< FUNCTION: erl2lin_problem >-----------------------------------
// Purpose:    constructs a linear problem from feature/target vectors
// Parameters: env     - current erlang environment
//             x       - training feature vector list
//             y       - list of target class labels
//             params  - additional problem parameters
//             problem - return the linear problem via here
// Returns:    pointer to problem
---------------------------------------------------------------------------*/
void erl2lin_problem (
   ErlNifEnv*      env,
   ERL_NIF_TERM    x,
   ERL_NIF_TERM    y,
   ERL_NIF_TERM    params,
   LINEAR_PROBLEM* problem)
{
   ERL_NIF_TERM key;
   ERL_NIF_TERM value;
   ERL_NIF_TERM head;
   ERL_NIF_TERM tail;
   ErlNifBinary vector;
   // get bias value
   key = enif_make_atom(env, "bias");
   CHECK(enif_get_map_value(env, params, key, &value), "missing_bias");
   CHECK(enif_get_double(env, value, &problem->bias), "invalid_bias");
   // get sample matrix size
   unsigned m;
   CHECK(enif_get_list_length(env, x, &m), "invalid_x");
   CHECK(enif_get_list_cell(env, x, &head, &tail), "missing_features");
   problem->l = m;
   // get feature vector size
   CHECK(enif_inspect_binary(env, head, &vector), "invalid_features");
   int n = vector.size / sizeof(float);
   problem->n = problem->bias < 0 ? n : n + 1;
   // copy feature/target values
   problem->x = erl2lin_features(env, x, m, problem->bias);
   problem->y = erl2lin_targets(env, y, m);
}
/*-----------< FUNCTION: erl2lin_model >-------------------------------------
// Purpose:    constructs a linear model structure from a map representation
// Parameters: env    - current erlang environment
//             params - model parameters (constructed via nif_lin_export)
// Returns:    pointer to the allocated and constructed model
---------------------------------------------------------------------------*/
LINEAR_MODEL* erl2lin_model (ErlNifEnv* env, ERL_NIF_TERM params)
{
   LINEAR_MODEL* model = nif_alloc<LINEAR_MODEL>();
   try {
      ERL_NIF_TERM key;
      ERL_NIF_TERM value;
      ERL_NIF_TERM tail;
      ErlNifBinary vector;
      // extract model parameters
      erl2lin_params(env, params, &model->param, 0);
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
      // extract bias
      model->bias = -1;
      key = enif_make_atom(env, "intercept");
      if (enif_get_map_value(env, params, key, &value))
         if (enif_is_binary(env, value))
            model->bias = 1;
      // extract feature count
      key = enif_make_atom(env, "coef");
      CHECK(enif_get_map_value(env, params, key, &tail), "missing_coef");
      CHECK(enif_get_list_cell(env, tail, &value, &tail), "missing_coef");
      CHECK(enif_inspect_binary(env, value, &vector), "invalid_coef");
      model->nr_feature = vector.size / sizeof(float);
      // extract coefficients
      int model_count = model->nr_class == 2
         ? 1
         : model->nr_class;
      int weight_count = model->bias >= 0
         ? model->nr_feature + 1
         : model->nr_feature;
      model->w = nif_alloc<double>(model_count * weight_count);
      for (int i = 0; i < model_count; i++) {
         for (int j = 0; j < model->nr_feature; j++)
            model->w[j * model_count + i] = ((float*)vector.data)[j];
         if (i < model_count - 1) {
            CHECK(enif_get_list_cell(env, tail, &value, &tail), "missing_coef");
            CHECK(enif_inspect_binary(env, value, &vector), "invalid_coef");
         }
      }
      // extract intercepts
      if (model->bias >= 0) {
         key = enif_make_atom(env, "intercept");
         CHECK(enif_get_map_value(env, params, key, &value), "missing_intercept");
         CHECK(enif_inspect_binary(env, value, &vector), "invalid_intercept");
         for (int i = 0; i < model_count; i++)
            model->w[model->nr_feature * model_count + i] =
               ((float*)vector.data)[i];
      }
      // extract prob_a
      key = enif_make_atom(env, "prob_a");
      CHECK(enif_get_map_value(env, params, key, &value), "missing_prob_a");
      if (!enif_is_identical(value, enif_make_atom(env, "nil"))) {
         CHECK(enif_inspect_binary(env, value, &vector), "invalid_prob_a");
         model->prob_a = nif_alloc<double>(vector.size / sizeof(float));
         for (int i = 0; i < (int)(vector.size / sizeof(float)); i++)
            model->prob_a[i] = ((float*)vector.data)[i];
      }
      // extract prob_b
      key = enif_make_atom(env, "prob_b");
      CHECK(enif_get_map_value(env, params, key, &value), "missing_prob_b");
      if (!enif_is_identical(value, enif_make_atom(env, "nil"))) {
         CHECK(enif_inspect_binary(env, value, &vector), "invalid_prob_b");
         model->prob_b = nif_alloc<double>(vector.size / sizeof(float));
         for (int i = 0; i < (int)(vector.size / sizeof(float)); i++)
            model->prob_b[i] = ((float*)vector.data)[i];
      }
      return model;
   } catch (NifError& e) {
      erl2lin_free_model(model);
      throw;
   }
}
/*-----------< FUNCTION: erl2lin_params >------------------------------------
// Purpose:    constructs a linear param structure from an options map
// Parameters: env      - current erlang environment
//             options  - linear model options map
//             params   - returns linear parameters via here
//             training - true if the model is being trained
//                        false if we are loading it from a trained model
// Returns:    pointer to params
---------------------------------------------------------------------------*/
void erl2lin_params (
   ErlNifEnv*    env,
   ERL_NIF_TERM  options,
   LINEAR_PARAM* params,
   int           training) {
   ERL_NIF_TERM key;
   ERL_NIF_TERM value;
   // decode solver type
   key = enif_make_atom(env, "solver");
   CHECK(enif_get_map_value(env, options, key, &value), "missing_solver");
   if (enif_is_identical(value, enif_make_atom(env, "l2r_lr")))
      params->solver_type = L2R_LR;
   else if (enif_is_identical(value, enif_make_atom(env, "l2r_l2loss_svc_dual")))
      params->solver_type = L2R_L2LOSS_SVC_DUAL;
   else if (enif_is_identical(value, enif_make_atom(env, "l2r_l2loss_svc")))
      params->solver_type = L2R_L2LOSS_SVC;
   else if (enif_is_identical(value, enif_make_atom(env, "l2r_l1loss_svc_dual")))
      params->solver_type = L2R_L1LOSS_SVC_DUAL;
   else if (enif_is_identical(value, enif_make_atom(env, "mcsvm_cs")))
      params->solver_type = MCSVM_CS;
   else if (enif_is_identical(value, enif_make_atom(env, "l1r_l2loss_svc")))
      params->solver_type = L1R_L2LOSS_SVC;
   else if (enif_is_identical(value, enif_make_atom(env, "l1r_lr")))
      params->solver_type = L1R_LR;
   else if (enif_is_identical(value, enif_make_atom(env, "l2r_lr_dual")))
      params->solver_type = L2R_LR_DUAL;
   else if (enif_is_identical(value, enif_make_atom(env, "l2r_l2loss_svr")))
      params->solver_type = L2R_L2LOSS_SVR;
   else if (enif_is_identical(value, enif_make_atom(env, "l2r_l2loss_svr_dual")))
      params->solver_type = L2R_L2LOSS_SVR_DUAL;
   else if (enif_is_identical(value, enif_make_atom(env, "l2r_l1loss_svr_dual")))
      params->solver_type = L2R_L1LOSS_SVR_DUAL;
   else
      throw NifError("invalid_solver");
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
      // decode stopping criteria
      key = enif_make_atom(env, "epsilon");
      CHECK(enif_get_map_value(env, options, key, &value), "missing_epsilon");
      CHECK(enif_get_double(env, value, &params->eps));
      // decode SVR sensitivity
      key = enif_make_atom(env, "p");
      CHECK(enif_get_map_value(env, options, key, &value), "missing_p");
      CHECK(enif_get_double(env, value, &params->p));
   }
}
/*-----------< FUNCTION: erl2lin_features >----------------------------------
// Purpose:    converts a list of feature vectors to a linear sparse matrix
// Parameters: env  - current erlang environment
//             x    - list of feature vectors (floats)
//             m    - number of vectors in the feature matrix
//             bias - feature vector bias term
// Returns:    pointer to an array of sparse feature vectors
---------------------------------------------------------------------------*/
LINEAR_NODE** erl2lin_features (
   ErlNifEnv*   env,
   ERL_NIF_TERM x,
   unsigned     m,
   double       bias)
{
   LINEAR_NODE** nodes = nif_alloc<LINEAR_NODE*>(m);
   try {
      for (int i = 0; i < (int)m; i++) {
         ERL_NIF_TERM head;
         CHECK(enif_get_list_cell(env, x, &head, &x), "missing_features");
         nodes[i] = erl2lin_feature(env, head, bias);
      }
      return nodes;
   } catch (...) {
      for (int i = 0; i < (int)m; i++)
         nif_free(nodes[i]);
      nif_free(nodes);
      throw;
   }
}
/*-----------< FUNCTION: erl2lin_feature >-----------------------------------
// Purpose:    converts a feature vector to a linear sparse vector
// Parameters: env  - current erlang environment
//             x    - feature vector (floats)
//             bias - bias term
// Returns:    pointer to a sparse feature vector
---------------------------------------------------------------------------*/
LINEAR_NODE* erl2lin_feature (ErlNifEnv* env, ERL_NIF_TERM x, double bias)
{
   ErlNifBinary vector;
   CHECK(enif_inspect_binary(env, x, &vector), "invalid_feature");
   // copy the feature vector to the sparse array
   int n = vector.size / sizeof(float);
   LINEAR_NODE* nodes = nif_alloc<LINEAR_NODE>(bias >= 0 ? n + 2 : n + 1);
   int j = 0;
   while (j < n) {
      nodes[j] = (LINEAR_NODE){
         .index = j + 1,
         .value = ((float*)vector.data)[j]
      };
      j++;
   }
   // add the bias term if specified
   if (bias >= 0) {
      nodes[j] = (LINEAR_NODE){ .index = j + 1, .value = bias };
      j++;
   }
   // terminate the sparse vector with -1 per liblinear spec
   nodes[j] = (LINEAR_NODE){ .index = -1, .value = 0 };
   return nodes;
}
/*-----------< FUNCTION: erl2lin_targets >-----------------------------------
// Purpose:    converts a list of target labels to an array of labels
// Parameters: env - current erlang environment
//             y   - list of target labels (integers)
//             m   - number of training labels
// Returns:    array of doubles representing the target labels
---------------------------------------------------------------------------*/
double* erl2lin_targets (ErlNifEnv* env, ERL_NIF_TERM y, unsigned m)
{
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
/*-----------< FUNCTION: erl2lin_free_problem >------------------------------
// Purpose:    frees the memory associated with a linear problem structure
// Parameters: problem - structure to free
// Returns:    none
---------------------------------------------------------------------------*/
void erl2lin_free_problem (LINEAR_PROBLEM* problem)
{
   if (problem->x)
      for (int i = 0; i < problem->l; i++)
         nif_free(problem->x[i]);
   nif_free(problem->x);
   problem->x = NULL;
   nif_free(problem->y);
   problem->y = NULL;
}
/*-----------< FUNCTION: erl2lin_free_params >-------------------------------
// Purpose:    frees the memory associated with a linear parameter structure
// Parameters: params - structure to free
// Returns:    none
---------------------------------------------------------------------------*/
void erl2lin_free_params (LINEAR_PARAM* params)
{
   nif_free(params->weight_label);
   params->weight_label = NULL;
   nif_free(params->weight);
   params->weight = NULL;
}
/*-----------< FUNCTION: lin2erl_model >-------------------------------------
// Purpose:    converts a linear model to an erlang map
// Parameters: env   - current erlang environment
//             model - linear model structure to convert
// Returns:    erlang map containing the model parameters
---------------------------------------------------------------------------*/
ERL_NIF_TERM lin2erl_model (ErlNifEnv* env, LINEAR_MODEL* model)
{
   ERL_NIF_TERM result = enif_make_new_map(env);
   ERL_NIF_TERM key;
   ERL_NIF_TERM value;
   ErlNifBinary vector;
   // encode version
   key   = enif_make_atom(env, "version");
   value = enif_make_int(env, 1);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode solver type
   key = enif_make_atom(env, "solver");
   switch (model->param.solver_type) {
      case L2R_LR:
         value = enif_make_atom(env, "l2r_lr");
         break;
      case L2R_L2LOSS_SVC_DUAL:
         value = enif_make_atom(env, "l2r_l2loss_svc_dual");
         break;
      case L2R_L2LOSS_SVC:
         value = enif_make_atom(env, "l2r_l2loss_svc");
         break;
      case L2R_L1LOSS_SVC_DUAL:
         value = enif_make_atom(env, "l2r_l1loss_svc_dual");
         break;
      case MCSVM_CS:
         value = enif_make_atom(env, "mcsvm_cs");
         break;
      case L1R_L2LOSS_SVC:
         value = enif_make_atom(env, "l1r_l2loss_svc");
         break;
      case L1R_LR:
         value = enif_make_atom(env, "l1r_lr");
         break;
      case L2R_LR_DUAL:
         value = enif_make_atom(env, "l2r_lr_dual");
         break;
      case L2R_L2LOSS_SVR:
         value = enif_make_atom(env, "l2r_l2loss_svr");
         break;
      case L2R_L2LOSS_SVR_DUAL:
         value = enif_make_atom(env, "l2r_l2loss_svr_dual");
         break;
      case L2R_L1LOSS_SVR_DUAL:
         value = enif_make_atom(env, "l2r_l1loss_svr_dual");
         break;
      default:
         value = enif_make_atom(env, "nil");
   }
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode classes
   ERL_NIF_TERM classes[model->nr_class]; memset(classes, 0, sizeof(classes));
   for (int i = 0; i < model->nr_class; i++)
      classes[i] = enif_make_int(env, model->label[i]);
   key   = enif_make_atom(env, "classes");
   value = enif_make_list_from_array(env, classes, model->nr_class);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode coefficients
   int model_count = model->nr_class == 2
      ? 1
      : model->nr_class;
   ERL_NIF_TERM coefs[model_count]; memset(coefs, 0, sizeof(coefs));
   for (int i = 0; i < model_count; i++) {
      int coef_count = model->nr_feature;
      CHECKALLOC(enif_alloc_binary(coef_count * sizeof(float), &vector));
      for (int j = 0; j < coef_count; j++)
         ((float*)vector.data)[j] = model->w[j * model_count + i];
      coefs[i] = enif_make_binary(env, &vector);
   }
   key   = enif_make_atom(env, "coef");
   value = enif_make_list_from_array(env, coefs, model_count);
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode intercepts
   key   = enif_make_atom(env, "intercept");
   value = enif_make_double(env, 0);
   if (model->bias >= 0) {
      CHECKALLOC(enif_alloc_binary(model_count * sizeof(float), &vector));
      for (int i = 0; i < model_count; i++)
         ((float*)vector.data)[i] =
            model->w[model->nr_feature * model_count + i];
      value = enif_make_binary(env, &vector);
   }
   CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   // encode prob_a
   key = enif_make_atom(env, "prob_a");
   if (model->prob_a) {
      CHECKALLOC(enif_alloc_binary(model_count * sizeof(float), &vector));
      for (int i = 0; i < model_count; i++)
         ((float*)vector.data)[i] = model->prob_a[i];
      value = enif_make_binary(env, &vector);
      CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   } else {
      value = enif_make_atom(env, "nil");
      CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   }
   // encode prob_b
   key = enif_make_atom(env, "prob_b");
   if (model->prob_b) {
      CHECKALLOC(enif_alloc_binary(model_count * sizeof(float), &vector));
      for (int i = 0; i < model_count; i++)
         ((float*)vector.data)[i] = model->prob_b[i];
      value = enif_make_binary(env, &vector);
      CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   } else {
      value = enif_make_atom(env, "nil");
      CHECKALLOC(enif_make_map_put(env, result, key, value, &result));
   }
   return result;
}
/*-----------< FUNCTION: lin2lin_model >-------------------------------------
// Purpose:    clones a linear model structure
// Parameters: source - linear model structure to copy
//             prob_a - calibration probability slope variables
//             prob_b - calibration probability intercept variables
// Returns:    cloned linear model
---------------------------------------------------------------------------*/
LINEAR_MODEL* lin2lin_model (model* source, double* prob_a, double* prob_b)
{
   LINEAR_MODEL* target = nif_alloc<LINEAR_MODEL>();
   try {
      // copy scalar fields + clear class weights (training-only)
      memcpy(&target->param, &source->param, sizeof(LINEAR_PARAM));
      target->nr_class           = source->nr_class;
      target->nr_feature         = source->nr_feature;
      target->bias               = source->bias;
      target->param.weight_label = NULL;
      target->param.weight       = NULL;
      target->prob_a             = prob_a;
      target->prob_b             = prob_b;
      // copy weights
      int model_count = source->nr_class == 2
         ? 1
         : source->nr_class;
      int weight_count = source->bias >= 0
         ? source->nr_feature + 1
         : source->nr_feature;
      target->w = nif_clone(source->w, model_count * weight_count);
      // copy labels
      target->label = nif_clone(source->label, target->nr_class);
   } catch (...) {
      erl2lin_free_model(target);
      throw;
   }
   return target;
}
/*-----------< FUNCTION: erl2lin_free_model >--------------------------------
// Purpose:    frees the memory associated with a linear model
// Parameters: model - linear model structure to free
// Returns:    none
---------------------------------------------------------------------------*/
void erl2lin_free_model (LINEAR_MODEL* model)
{
   if (model != NULL) {
      erl2lin_free_params(&model->param);
      nif_free(model->w);
      nif_free(model->label);
      nif_free(model->prob_a);
      nif_free(model->prob_b);
   }
   nif_free(model);
}
/*-----------< FUNCTION: lin_print >-----------------------------------------
// Purpose:    liblinear debug output callback
// Parameters: message - message to display
// Returns:    none
---------------------------------------------------------------------------*/
void lin_print (const char* message) {
   // suppress debug output
}
/*-----------< FUNCTION: lin_calibrate_train >-------------------------------
// Purpose:    calibrates decision outputs to class probabilities using
//             univariate binary logistic regression
//             lifted from libsvm sigmoid_train, which is not exported
// Parameters: m        - number of training examples
//             decision - decision ouputs for each training example
//                        used as X in the calibration model
//             labels   - class labels for each training example (pos/neg)
//             prob_a   - return regression slope parameter via here
//             prob_b   - return regression intercept parameter via here
// Returns:    none
---------------------------------------------------------------------------*/
void lin_calibrate_train (
   int     m,
   double* decision,
   double* labels,
   double& prob_a,
   double& prob_b)
{
   double prior1=0, prior0 = 0;
   int i;

   for (i=0;i<m;i++)
      if (labels[i] > 0) prior1+=1;
      else prior0+=1;

   int max_iter=100; // Maximal number of iterations
   double min_step=1e-10;  // Minimal step taken in line search
   double sigma=1e-12;  // For numerically strict PD of Hessian
   double eps=1e-5;
   double hiTarget=(prior1+1.0)/(prior1+2.0);
   double loTarget=1/(prior0+2.0);
   double *t=nif_alloc<double>(m);
   double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
   double newA,newB,newf,d1,d2;
   int iter;

   // Initial Point and Initial Fun Value
   prob_a=0.0; prob_b=log((prior0+1.0)/(prior1+1.0));
   double fval = 0.0;

   for (i=0;i<m;i++)
   {
      if (labels[i]>0) t[i]=hiTarget;
      else t[i]=loTarget;
      fApB = decision[i]*prob_a+prob_b;
      if (fApB>=0)
         fval += t[i]*fApB + log(1+exp(-fApB));
      else
         fval += (t[i] - 1)*fApB +log(1+exp(fApB));
   }
   for (iter=0;iter<max_iter;iter++)
   {
      // Update Gradient and Hessian (use H' = H + sigma I)
      h11=sigma; // numerically ensures strict PD
      h22=sigma;
      h21=0.0;g1=0.0;g2=0.0;
      for (i=0;i<m;i++)
      {
         fApB = decision[i]*prob_a+prob_b;
         if (fApB >= 0)
         {
            p=exp(-fApB)/(1.0+exp(-fApB));
            q=1.0/(1.0+exp(-fApB));
         }
         else
         {
            p=1.0/(1.0+exp(fApB));
            q=exp(fApB)/(1.0+exp(fApB));
         }
         d2=p*q;
         h11+=decision[i]*decision[i]*d2;
         h22+=d2;
         h21+=decision[i]*d2;
         d1=t[i]-p;
         g1+=decision[i]*d1;
         g2+=d1;
      }

      // Stopping Criteria
      if (fabs(g1)<eps && fabs(g2)<eps)
         break;

      // Finding Newton direction: -inv(H') * g
      det=h11*h22-h21*h21;
      dA=-(h22*g1 - h21 * g2) / det;
      dB=-(-h21*g1+ h11 * g2) / det;
      gd=g1*dA+g2*dB;


      stepsize = 1;     // Line Search
      while (stepsize >= min_step)
      {
         newA = prob_a + stepsize * dA;
         newB = prob_b + stepsize * dB;

         // New function value
         newf = 0.0;
         for (i=0;i<m;i++)
         {
            fApB = decision[i]*newA+newB;
            if (fApB >= 0)
               newf += t[i]*fApB + log(1+exp(-fApB));
            else
               newf += (t[i] - 1)*fApB +log(1+exp(fApB));
         }
         // Check sufficient decrease
         if (newf<fval+0.0001*stepsize*gd)
         {
            prob_a=newA;prob_b=newB;fval=newf;
            break;
         }
         else
            stepsize = stepsize / 2.0;
      }

      CHECK(stepsize >= min_step, "calibration line search failed");
   }

   CHECK(iter < max_iter, "exceeded calibration max iterations");
   nif_free(t);
}
/*-----------< FUNCTION: lin_calibrate_predict >-----------------------------
// Purpose:    produces a calibrated probability estimate for a
//             non-probabilistic classifier (like svm)
//             lifted from libsvm sigmoid_predict, which is not exported
// Parameters: decision - decision output, used as x in the calibration model
//             prob_a   - regression slope parameter
//             prob_b   - regression intercept parameter
// Returns:    none
---------------------------------------------------------------------------*/
double lin_calibrate_predict (
   double decision,
   double prob_a,
   double prob_b)
{
   double fApB = decision*prob_a+prob_b;
   // 1-p used later; avoid catastrophic cancellation
   if (fApB >= 0)
      return exp(-fApB)/(1.0+exp(-fApB));
   else
      return 1.0/(1+exp(fApB)) ;
}
