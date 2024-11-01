#pragma once

#include "UnitTest.h"
#include <QObject>
#include <QCoreapplication>
#include "NeuralNet.h"




class TST_BackpropagationXOR : public UnitTest::Test
{
	TEST_CLASS(TST_BackpropagationXOR)
public:
	TST_BackpropagationXOR()
		: Test("TST_BackpropagationXOR")
	{
		ADD_TEST(TST_BackpropagationXOR::test1);
		//ADD_TEST(test2);

	}

private:

	// Tests
	TEST_FUNCTION(test1)
	{
		TEST_START(results);

		int a = 0;
		TEST_MESSAGE("is a == 0?");
		TEST_ASSERT(a == 0);

		// Setup traing set
		struct TrainingSample
		{
			std::vector<float> inputs;
			std::vector<float> expectedOutput;
		};
		std::vector<TrainingSample> m_trainingData;
		m_trainingData.push_back({ {0,0},{0} });
		m_trainingData.push_back({ {0,1},{1} });
		m_trainingData.push_back({ {1,0},{1} });
		m_trainingData.push_back({ {1,1},{0} });

		// Setup net
		unsigned int inputs = 2;
		unsigned int outputs = 1;
		unsigned int hiddenLayers = 1;
		unsigned int hiddenLayerSize = 2;
		if (m_trainingData.size() > 0)
		{
			inputs = m_trainingData[0].inputs.size();
			outputs = m_trainingData[0].expectedOutput.size();
		}
		std::vector < NeuralNet::Neuron::ID> inputIDs(inputs, 0);
		std::vector < NeuralNet::Neuron::ID> outputIDs(outputs, 0);
		for (size_t i = 0; i < inputIDs.size(); ++i)
			inputIDs[i] = i;
		for (size_t i = 0; i < outputIDs.size(); ++i)
			outputIDs[i] = inputIDs.size() + i;

		NeuralNet::FullConnectedNeuralNet net(inputIDs, hiddenLayers, hiddenLayerSize, outputIDs);

		net.setActivationType(NeuralNet::Activation::Type::tanh_);
		net.setActivationType(outputIDs[0], NeuralNet::Activation::Type::sigmoid);
		if(hiddenLayerSize >= 3)
			net.setActivationType(net.getNeuron(1, 2)->getID(), NeuralNet::Activation::Type::gaussian);

		TEST_ASSERT(net.getInputSize() == inputs);
		TEST_ASSERT(net.getOutputSize() == outputs);


		// Start training
		NeuralNet::LearnAlgo::Backpropagation::setLearningRate(1);

		bool doTrain = true;
		size_t iteration = 0;
		size_t maxIterations = 100000;
		double elapsedTime = 0;
		auto start = std::chrono::high_resolution_clock::now();
		while(doTrain)
		{
			float netError = 0;
			for (const TrainingSample& sample : m_trainingData)
			{
				net.setInputValues(sample.inputs);
				net.update();
				NeuralNet::LearnAlgo::Backpropagation::learn(net, sample.expectedOutput);
				netError += NeuralNet::LearnAlgo::Backpropagation::getNetError(net,sample.expectedOutput);
			}
			
			netError /= m_trainingData.size();
			++iteration;
			if (iteration % 1000 == 0)
			{
				auto end = std::chrono::high_resolution_clock::now();
				elapsedTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				TEST_MESSAGE("Iteration: " +std::to_string(iteration) + " Error: " +std::to_string(netError));
				start = std::chrono::high_resolution_clock::now();
			}
			if (netError < 0.01)
			{
				auto end = std::chrono::high_resolution_clock::now();
				elapsedTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				TEST_MESSAGE("Iteration: " + std::to_string(iteration) + " Error: " + std::to_string(netError));
				TEST_MESSAGE("Training resolved in a successful solution after: " + std::to_string(elapsedTime) + "ms");
				//TEST_ASSERT(true);
				doTrain = false;
			}
			if (iteration >= maxIterations)
			{
				TEST_MESSAGE("Iteration: " + std::to_string(iteration) + " Error: " + std::to_string(netError));
				TEST_ASSERT_M(false, "Training did not resolve in a successful solution");
			}
		}

		// Test net
		for (const TrainingSample& sample : m_trainingData)
		{
			net.setInputValues(sample.inputs);
			net.update();
			std::vector<float> output = net.getOutputValues();
			std::string msg = "Input: ";
			for (float f : sample.inputs)
				msg += std::to_string(f) + " ";
			msg += "Output: ";
			for (float f : output)
				msg += std::to_string(f) + " ";
			msg += "Expected: ";
			for (float f : sample.expectedOutput)
				msg += std::to_string(f) + " ";
			TEST_MESSAGE(msg);
		}

		// Print genom
		std::vector<float> weights = net.getWeights();
		std::vector<float> bias = net.getBias();
		std::string msg = "Weights: ";
		for (float f : weights)
			msg += std::to_string(f) + " ";
		msg += "Bias: ";
		for (float f : bias)
			msg += std::to_string(f) + " ";
		TEST_MESSAGE(msg);
	}
};
