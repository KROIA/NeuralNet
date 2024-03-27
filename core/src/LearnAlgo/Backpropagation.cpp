#include "LearnAlgo/Backpropagation.h"
#include <unordered_map>

namespace NeuralNet
{
	namespace LearnAlgo
	{
		Backpropagation::Backpropagation()
		{

		}
		Backpropagation::~Backpropagation()
		{

		}

		void Backpropagation::learn(std::vector<Layer>& layers, const std::vector<float>& expectedOutput)
		{
			// NN_UNUSED(layers);
			// NN_UNUSED(expectedOutput);

			//Layer& outputLayer = layers.back();

			std::unordered_map<Neuron*, float> neuronError;
			for (int i = 0; i < layers.size(); ++i)
			{
				Layer& layer = layers[i];
				for (int j = 0; j < layer.neurons.size(); ++j)
				{
					Neuron* neuron = layer.neurons[j];
					neuronError[neuron] = 0;
				}
			}

			
			/*std::vector<float> layerError(layers[layers.size() - 1].neurons.size(), 0);
			size_t nextLayerErrorSize = 0;
			if (layers.size() > 2)
			{
				nextLayerErrorSize = layers[layers.size() - 2].neurons.size();
			}
			else
			{
				
			}*/
			//std::vector<float> nextLayerError(nextLayerErrorSize, 0);
			Layer& currentLayer = layers[layers.size() - 1];
			for (size_t i = 0; i < currentLayer.neurons.size(); i++)
			{
				Neuron* neuron = currentLayer.neurons[i];
				//Activation::ActivationFunction deriv = Activation::getActivationDerivetiveFunction(neuron->getActivationType());
				float output = neuron->getOutput();
				float error = getError(output, expectedOutput[i]);
				neuronError[neuron] = error;
			}

			for (int layerIdx = layers.size() - 1; layerIdx > 0; --layerIdx)
			{
				Layer &activeLayer = layers[layerIdx];
				for (size_t i = 0; i < activeLayer.neurons.size(); i++)
				{
					Neuron* neuron = activeLayer.neurons[i];
					Activation::ActivationFunction deriv = Activation::getActivationDerivetiveFunction(neuron->getActivationType());
					std::vector<Connection*> connections = neuron->getInputConnections();

					float error = deriv(neuron->getNetInput()) * neuronError[neuron];

					for (size_t y = 0; y < connections.size(); ++y)
					{
						Neuron* startNeuron = connections[y]->getStartNeuron();
						if(layerIdx > 1)
							neuronError[startNeuron] += error * connections[y]->getWeight();
						
						float deltaW = startNeuron->getOutput() * error * m_learningRage;
						connections[y]->updateWeight(deltaW);
					}
				}
				//layerError = nextLayerError;
				//nextLayerErrorSize = layers[layerIdx - 1].neurons.size();
				//nextLayerError = std::vector<float>(nextLayerErrorSize, 0);
			}
		}

		float Backpropagation::getError(float output, float expectedOutput)
		{
			return expectedOutput - output;
		}
	}
}