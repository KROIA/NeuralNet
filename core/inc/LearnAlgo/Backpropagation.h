#pragma once

#include "NeuralNet_base.h"
#include <vector>
#include "SimpleImpl/NetworkComponents/Layer.h"

namespace NeuralNet
{
	namespace LearnAlgo
	{
		class NEURAL_NET_EXPORT Backpropagation
		{
		public:

			Backpropagation();
			~Backpropagation();

			void setLearningRate(float learningRate){ m_learningRate = learningRate;}
			float getLearningRate() const { return m_learningRate; }

			void learn(std::vector<Layer>& layers, const std::vector<float> &expectedOutput);
		
			static float getError(float output, float expectedOutput);

		protected:

		private:
			float m_learningRate = 1.f;
		};
	}
}