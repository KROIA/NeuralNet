// @file NeuralNet.h
// @brief Main public header for the library.
//
// Include this single header to access the entire public API.
// Add your own public headers inside USER_SECTION 2 so that
// consumers only need `#include "NeuralNet.h"`.
#pragma once

/// USER_SECTION_START 1

/// USER_SECTION_END

#include "NeuralNet_info.h"

/// USER_SECTION_START 2
#include "Base/ActivationFunction.h"
#include "Base/NeuralNetBase.h"

#include "SimpleImpl/Nets/FullConnectedNeuralNet.h"
#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"
#include "SimpleImpl/NetworkComponents/Neuron.h"
#include "SimpleImpl/NetworkComponents/InputNeuron.h"
#include "SimpleImpl/NetworkComponents/Connection.h"


#include "Visualisation/CustomConnectedNeuralNetPainter.h"
#include "Visualisation/NeuralNetCanvasObject.h"

#include "LearnAlgo/Backpropagation.h"
#include "LearnAlgo/GeneticLearn.h"

/// USER_SECTION_END