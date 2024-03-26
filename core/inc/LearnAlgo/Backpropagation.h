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

			void learn(std::vector<Layer>& layers, const std::vector<float> &expectedOutput);
		
			static float getError(float output, float expectedOutput);

		protected:

		private:
			float m_learningRage = 1.f;
		};
	}
}