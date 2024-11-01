#pragma once

#include "UnitTest.h"
#include <QObject>
#include <QCoreapplication>
#include "NeuralNet.h"




class TST_GeneticLearnXOR : public UnitTest::Test
{
	TEST_CLASS(TST_GeneticLearnXOR)
public:
	TST_GeneticLearnXOR()
		: Test("TST_GeneticLearnXOR")
	{
		ADD_TEST(TST_GeneticLearnXOR::test1);
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
		unsigned int populationSize = 100;
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

		std::vector<NeuralNet::LearnAlgo::GeneticLearn::GeneticPerformance> nets;
		for (unsigned int i = 0; i < populationSize; ++i)
		{
			NeuralNet::FullConnectedNeuralNet* net = new NeuralNet::FullConnectedNeuralNet(inputIDs, hiddenLayers, hiddenLayerSize, outputIDs);
			net->setActivationType(NeuralNet::Activation::Type::tanh_);
			net->setActivationType(outputIDs[0], NeuralNet::Activation::Type::sigmoid);
			if (hiddenLayerSize >= 3)
				net->setActivationType(net->getNeuron(1, 2)->getID(), NeuralNet::Activation::Type::gaussian);
			nets.push_back({ net, 0 });
		}

		// Start training
		NeuralNet::LearnAlgo::GeneticLearn::setCrossoverCountPercentage(50);
		NeuralNet::LearnAlgo::GeneticLearn::setMutationCountPercentage(50);
		NeuralNet::LearnAlgo::GeneticLearn::setMutationRate(0.01f);

		bool doTrain = true;
		size_t iteration = 0;
		size_t maxIterations = 10000;
		double elapsedTime = 0;
		auto start = std::chrono::high_resolution_clock::now();
		while (doTrain)
		{
			float averageError = 0;
			for (NeuralNet::LearnAlgo::GeneticLearn::GeneticPerformance& gp : nets)
			{
				float netError = 0;
				for (const TrainingSample& sample : m_trainingData)
				{

					gp.net->setInputValues(sample.inputs);
					gp.net->update();
					std::vector<float> output = gp.net->getOutputValues();
					float error = 0;
					for (size_t i = 0; i < output.size(); ++i)
					{
						error += abs(output[i] - sample.expectedOutput[i]);
					}
					netError += error;
				}
				gp.fitness = m_trainingData.size() / netError;
				averageError += netError;
			}
			NeuralNet::LearnAlgo::GeneticLearn::learnAndReplace(nets);
			

			averageError /= nets.size();
			++iteration;
			if (iteration % 1000 == 0)
			{
				auto end = std::chrono::high_resolution_clock::now();
				elapsedTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				TEST_MESSAGE("Iteration: " + std::to_string(iteration) + " Error: " + std::to_string(averageError));
				start = std::chrono::high_resolution_clock::now();
			}
			if (averageError < 0.01)
			{
				auto end = std::chrono::high_resolution_clock::now();
				elapsedTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				TEST_MESSAGE("Iteration: " + std::to_string(iteration) + " Error: " + std::to_string(averageError));
				TEST_MESSAGE("Training resolved in a successful solution after: " + std::to_string(elapsedTime) + "ms");
				//TEST_ASSERT(true);
				doTrain = false;
			}
			if (iteration >= maxIterations)
			{
				TEST_MESSAGE("Iteration: " + std::to_string(iteration) + " Error: " + std::to_string(averageError));
				TEST_ASSERT_M(false, "Training did not resolve in a successful solution");
			}
		}

		// Test net
		// Find the best net
		NeuralNet::LearnAlgo::GeneticLearn::GeneticPerformance bestNet = nets[0];
		for (NeuralNet::LearnAlgo::GeneticLearn::GeneticPerformance& gp : nets)
		{
			if (gp.fitness > bestNet.fitness)
				bestNet = gp;
		}

		for (const TrainingSample& sample : m_trainingData)
		{
			bestNet.net->setInputValues(sample.inputs);
			bestNet.net->update();
			std::vector<float> output = bestNet.net->getOutputValues();
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
		std::vector<float> weights = bestNet.net->getWeights();
		std::vector<float> bias = bestNet.net->getBias();
		std::string msg = "Weights: ";
		for (float f : weights)
			msg += std::to_string(f) + " ";
		msg += "Bias: ";
		for (float f : bias)
			msg += std::to_string(f) + " ";
		TEST_MESSAGE(msg);
	}
};
