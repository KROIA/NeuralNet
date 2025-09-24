#pragma once

#include "NeuralNet_base.h"
#include <vector>
#include "SimpleImpl/NetworkComponents/Layer.h"
#include "SimpleImpl/Nets/FullConnectedNeuralNet.h"
#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"

namespace NeuralNet
{
	namespace LearnAlgo
	{
		class NEURAL_NET_API Backpropagation
		{
			Backpropagation() {}
		public:

			

			static void setLearningRate(float learningRate){ m_learningRate = learningRate;}
			static float getLearningRate() { return m_learningRate; }

			static void learn(std::vector<Layer>& layers, const std::vector<float> &expectedOutput);
			//static void learn(FullConnectedNeuralNet& nn, const std::vector<float> &expectedOutput);
			static void learn(CustomConnectedNeuralNet& nn, const std::vector<float> &expectedOutput);
		
			//static std::vector<float> getOutputError(FullConnectedNeuralNet& nn, const std::vector<float>& expectedOutput);
			static std::vector<float> getOutputError(CustomConnectedNeuralNet& nn, const std::vector<float>& expectedOutput);

			//static float getNetError(FullConnectedNeuralNet& nn, const std::vector<float>& expectedOutput);
			static float getNetError(CustomConnectedNeuralNet& nn, const std::vector<float>& expectedOutput);

			static float getError(float output, float expectedOutput);

		protected:

		private:
			static float m_learningRate;
		};
	}
}