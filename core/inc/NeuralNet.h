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

/// USER_SECTION_END