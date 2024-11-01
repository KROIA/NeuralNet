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
		ADD_TEST(TST_FullConnectedNeuralNet::smallNet);

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
	TEST_FUNCTION(smallNet)
	{
		TEST_START;

		std::vector < NeuralNet::Neuron::ID> inputIds = { 0, 1 };
		std::vector < NeuralNet::Neuron::ID> outputIds = { 3 };
		unsigned int hiddenLayerCount = 1;
		unsigned int hiddenLayerSize = 1;
		NeuralNet::FullConnectedNeuralNet net(inputIds, hiddenLayerCount, hiddenLayerSize, outputIds);
		net.setActivationType(NeuralNet::Activation::Type::linear);
		std::vector<float> weights = net.getWeights();
		for (size_t i = 0; i < weights.size(); i++)
		{
			weights[i] = 1.f;
		}
		net.setBias(0.f);
		net.setWeights(weights);
		
		TEST_ASSERT(compare(net.getInputValues(), std::vector<float>(inputIds.size(), 0)));
		net.setInputValues(std::vector<float>(inputIds.size(), 1));
		TEST_ASSERT(compare(net.getOutputValues(), std::vector<float>(outputIds.size(), 0)));
		net.update();
		std::vector<float> outputs = net.getOutputValues();
		TEST_ASSERT(compare(outputs, std::vector<float>(outputIds.size(), 2.f)));
		net.setInputValues({0.f, 1.f});
		net.update();
		TEST_ASSERT(compare(net.getOutputValues(), std::vector<float>(outputIds.size(), 1.f)));
	}
};
