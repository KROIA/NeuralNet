#pragma once

#include "UnitTest.h"
#include <QObject>
#include <QCoreapplication>
#include "NeuralNet.h"




class TST_FullConnectedNeuralNet : public UnitTest::Test
{
	TEST_CLASS(TST_FullConnectedNeuralNet)
public:
	TST_FullConnectedNeuralNet()
		: Test("TST_FullConnectedNeuralNet")
	{
		ADD_TEST(TST_FullConnectedNeuralNet::smalNet);

	}

private:
	float m_tollerance = 0.0001f;
	bool compare(float a, float b)
	{
		return abs(a - b) < m_tollerance;
	}

	bool compare(const std::vector<float>& a, const std::vector<float>& b)
	{
		if (a.size() != b.size())
			return false;

		for (unsigned int i = 0; i < a.size(); i++)
		{
			if (!compare(a[i], b[i]))
				return false;
		}

		return true;
	}

	// Tests
	TEST_FUNCTION(smalNet)
	{
		TEST_START;

		unsigned int inputSize = 2;
		unsigned int outputSize = 1;
		unsigned int hiddenLayerCount = 1;
		unsigned int hiddenLayerSize = 1;
		NeuralNet::FullConnectedNeuralNet net(inputSize, hiddenLayerCount, hiddenLayerSize, outputSize);
		net.setActivationType(NeuralNet::Activation::Type::linear);
		std::vector<float> weights = net.getWeights();
		for (size_t i = 0; i < weights.size(); i++)
		{
			weights[i] = 1.f;
		}
		net.setWeights(weights);
		
		TEST_ASSERT(compare(net.getOutputValues(), std::vector<float>(outputSize, 0)));
		net.setInputValues(std::vector<float>(inputSize, 1));
		TEST_ASSERT(compare(net.getOutputValues(), std::vector<float>(outputSize, 0)));
		net.update();
		std::vector<float> outputs = net.getOutputValues();
		TEST_ASSERT(compare(outputs, std::vector<float>(outputSize, 2.f)));
		net.setInputValues({0.f, 1.f});
		net.update();
		TEST_ASSERT(compare(net.getOutputValues(), std::vector<float>(outputSize, 1.f)));
	}
};
