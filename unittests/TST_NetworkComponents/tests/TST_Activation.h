#pragma once

#include "UnitTest.h"
#include <QObject>
#include <QCoreapplication>
#include <vector>

#include "NeuralNet.h"



class TST_Activation : public UnitTest::Test
{
	TEST_CLASS(TST_Activation)
public:
	TST_Activation()
		: Test("TST_Activation")
	{
		ADD_TEST(TST_Activation::checkFunctions);
		ADD_TEST(TST_Activation::checkTestSets);
		buildTestSets();

	}

private:
	float m_tollerance = 0.0001f;
	struct TestSet
	{
		struct InputOutputPair
		{
			float input;
			float output;
		};

		NeuralNet::Activation::Type type;
		std::vector<InputOutputPair> testCases;
	};
	std::vector<TestSet> m_testSets;
	std::vector<TestSet> m_testSetsDerivetives;

	void buildTestSets()
	{
		size_t numTestCases = 100;
		m_testSets.resize(NeuralNet::Activation::Type::count);
		m_testSetsDerivetives.resize(NeuralNet::Activation::Type::count);

		for(size_t i = 0; i < NeuralNet::Activation::Type::count; ++i)
		{
			m_testSets[i].type = static_cast<NeuralNet::Activation::Type>(i);
			m_testSets[i].testCases.resize(numTestCases);

			m_testSetsDerivetives[i].type = static_cast<NeuralNet::Activation::Type>(i);
			m_testSetsDerivetives[i].testCases.resize(numTestCases);

			for (size_t j = 0; j < numTestCases; ++j)
			{
				float input = (rand() % 2000) / 100.0f - 10.f;
				m_testSets[i].testCases[j].input = input;
				m_testSetsDerivetives[i].testCases[j].input = input;
				float output = 0;
				float derivative = 0;
				switch (m_testSets[i].type)
				{
				case NeuralNet::Activation::Type::linear:		output = input;		derivative = 1;  break;
				case NeuralNet::Activation::Type::finiteLinear:	output = (input < -1.f ? -1.f : (input > 1.f ? 1.f : input));	derivative = (input < -1.f ? 0.f : (input > 1.f ? 0.f : 1.f));	break;
					case NeuralNet::Activation::Type::relu:			output = (input>0.f?input:0.f);		derivative = (input > 0.f ? 1 : 0.f);	break;
					case NeuralNet::Activation::Type::binary:		output = (input>0?1.f:0.f);		derivative = (input == 0? 1:0); break;
					case NeuralNet::Activation::Type::sigmoid:	
					{
						output = 1.f / (1.f + exp(-input));
						float ex = exp(input);
						float sum = ex + 1;
						derivative = ex / (sum * sum);
						break;
					}
					case NeuralNet::Activation::Type::gaussian:		output = exp(-input * input);		derivative = -2*exp(-input*input)*input; break;
					case NeuralNet::Activation::Type::tanh_:	
					{
						output = tanh(input);		
						float cosh_ = cosh(input);
						derivative = 1.f / (cosh_* cosh_);
						break;
					}
				default:
					std::cout << "Unknown activation type\n";
				}
				m_testSets[i].testCases[j].output = output;
				m_testSetsDerivetives[i].testCases[j].output = derivative;
			}
		}
	}


	// Tests
	bool checkFunctions(TestResults& results)
	{
		TEST_START(results);

		for(auto & testSet : m_testSets)
		{
			NeuralNet::Activation::ActivationFunction function = NeuralNet::Activation::getActivationFunction(testSet.type);
			TEST_ASSERT_M(function != nullptr, "Activationfunction for type: "+std::to_string(testSet.type)+" is not defined");
		}
		for (auto& testSet : m_testSetsDerivetives)
		{
			NeuralNet::Activation::ActivationFunction functionDeriv = NeuralNet::Activation::getActivationDerivetiveFunction(testSet.type);
			TEST_ASSERT_M(functionDeriv != nullptr, "Activationfunction derivative for type: " + std::to_string(testSet.type) + " is not defined");
		}

		TEST_END;
	}




	bool checkTestSets(TestResults& results)
	{
		TEST_START(results);

		for(auto & testSet : m_testSets)
		{
			checkSet(results, testSet);
		}

		for (auto& testSet : m_testSetsDerivetives)
		{
			checkSetDerivative(results, testSet);
		}


		TEST_END;
	}

	bool checkSet(TestResults & results, const TestSet& testSet)
	{
		TEST_START(results);
		NeuralNet::Activation::ActivationFunction function = NeuralNet::Activation::getActivationFunction(testSet.type);

		TEST_MESSAGE("Testing activation function: " + NeuralNet::Activation::getActivationName(testSet.type));
		for (const auto& testCase : testSet.testCases)
		{
			float output = function(testCase.input);
			float difference = output - testCase.output;
			TEST_ASSERT_M(abs(difference) < m_tollerance, 
				"Difference between expected and actual output is too large: " + std::to_string(difference));
		}

		TEST_END;
	}
	bool checkSetDerivative(TestResults& results, const TestSet& testSet)
	{
		TEST_START(results);
		NeuralNet::Activation::ActivationFunction function = NeuralNet::Activation::getActivationDerivetiveFunction(testSet.type);

		TEST_MESSAGE("Testing activation derivative function: " + NeuralNet::Activation::getActivationName(testSet.type));
		for (const auto& testCase : testSet.testCases)
		{
			float output = function(testCase.input);
			float difference = output - testCase.output;
			TEST_ASSERT_M(abs(difference) < m_tollerance,
				"Difference between expected and actual output is too large: " + std::to_string(difference));
		}

		TEST_END;
	}

};
