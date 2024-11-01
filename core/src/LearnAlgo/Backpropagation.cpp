#include "LearnAlgo/Backpropagation.h"
#include <unordered_map>

namespace NeuralNet
{
	namespace LearnAlgo
	{
		float Backpropagation::m_learningRate = 1;

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

					float netInput = neuron->getNetInput();
					if(neuron->isNetinputNormalizedEnabled())
					{
						netInput *= neuron->getInputConnections().size() + 1;
					}
					float error = deriv(netInput) * neuronError[neuron];

					for (size_t y = 0; y < connections.size(); ++y)
					{
						Neuron* startNeuron = connections[y]->getStartNeuron();
						if(layerIdx > 1)
							neuronError[startNeuron] += error * connections[y]->getWeight();
						
						float deltaW = startNeuron->getOutput() * error * m_learningRate;
						connections[y]->updateWeight(deltaW);
					}
					float deltaBias = error * m_learningRate;
					neuron->setBias(neuron->getBias() + deltaBias);
				}
				//layerError = nextLayerError;
				//nextLayerErrorSize = layers[layerIdx - 1].neurons.size();
				//nextLayerError = std::vector<float>(nextLayerErrorSize, 0);
			}
		}
		/*void Backpropagation::learn(FullConnectedNeuralNet& nn, const std::vector<float>& expectedOutput)
		{
			NN_UNUSED(nn);
			NN_UNUSED(expectedOutput);
			//learn(nn.getNetworkData().layers, expectedOutput);
		}*/
		void Backpropagation::learn(CustomConnectedNeuralNet& nn, const std::vector<float>& expectedOutput)
		{
			learn(nn.getNetworkData().layers, expectedOutput);
		}

		/*std::vector<float> Backpropagation::getOutputError(FullConnectedNeuralNet& nn, const std::vector<float>& expectedOutput)
		{
			if (expectedOutput.size() != nn.getOutputSize())
				return std::vector<float>();
			std::vector<float> err(nn.getOutputSize(), 0);
			for (size_t i = 0; i < nn.getOutputSize(); ++i)
			{
				err[i] = getError(nn.getOutputValue(i), expectedOutput[i]);
			}
			return err;
		}*/
		std::vector<float> Backpropagation::getOutputError(CustomConnectedNeuralNet& nn, const std::vector<float>& expectedOutput)
		{
			if (expectedOutput.size() != nn.getOutputSize())
				return std::vector<float>();
			std::vector<float> err(nn.getOutputSize(), 0);
			for (size_t i = 0; i < nn.getOutputSize(); ++i)
			{
				err[i] = getError(nn.getOutputValue(i), expectedOutput[i]);
			}
			return err;
		}

		/*float Backpropagation::getNetError(FullConnectedNeuralNet& nn, const std::vector<float>& expectedOutput)
		{
			if (expectedOutput.size() != nn.getOutputSize())
				return 0;
			float netError = 0;
			for (size_t i = 0; i < nn.getOutputSize(); ++i)
			{
				float outp = nn.getOutputValue(i);
				float expected = expectedOutput[i];
				float diff = getError(outp, expected);
				netError += std::abs(diff);
			}
			netError /= nn.getOutputSize();
			return netError;
		}*/
		float Backpropagation::getNetError(CustomConnectedNeuralNet& nn, const std::vector<float>& expectedOutput)
		{
			if (expectedOutput.size() != nn.getOutputSize())
				return 0;
			float netError = 0;
			for (size_t i = 0; i < nn.getOutputSize(); ++i)
			{
				float outp = nn.getOutputValue(i);
				float expected = expectedOutput[i];
				float diff = getError(outp, expected);
				netError += std::abs(diff);
			}
			netError /= nn.getOutputSize();
			return netError;
		}

		float Backpropagation::getError(float output, float expectedOutput)
		{
			return expectedOutput - output;
		}
	}
}