#pragma once

#include "UnitTest.h"
#include <QObject>
#include <QCoreapplication>
#include "NeuralNet.h"




class TST_Neuron : public UnitTest::Test
{
	TEST_CLASS(TST_Neuron)
public:
	TST_Neuron()
		: Test("TST_Neuron")
	{
		ADD_TEST(TST_Neuron::calcNetinput);
		ADD_TEST(TST_Neuron::calcOutput);

	}

private:
	float m_tollerance = 0.0001f;
	bool compare(float a, float b)
	{
		return abs(a - b) < m_tollerance;
	}

	// Tests
	bool calcNetinput(TestResults& results)
	{
		TEST_START(results);

		/*NeuralNet::Neuron neuron;
		neuron.addInputConnection(&connection);
		neuron.setActivationType(NeuralNet::Activation::Type::linear);
		TEST_ASSERT(compare(neuron.getNetInput(), 0), "Output shuld be 0");
		neuron.update();
		TEST_ASSERT(compare(neuron.getNetInput(), 0), "Output shuld be 0");

		connection.addInputValue(1.0f);
		neuron.update();
		TEST_ASSERT(compare(neuron.getNetInput(), 1));

		neuron.addInputValue(2.0f);
		neuron.update();
		TEST_ASSERT(compare(neuron.getNetInput(), 3));*/

		TEST_END;
	}




	bool calcOutput(TestResults& results)
	{
		TEST_START(results);

		/*NeuralNet::Neuron neuron;
		neuron.setActivationType(NeuralNet::Activation::Type::linear);
		neuron.addInputValue(1.0f);
		neuron.update();
		TEST_ASSERT(compare(neuron.getOutput(), 1));
		neuron.setActivationType(NeuralNet::Activation::Type::relu);
		neuron.update();
		TEST_ASSERT(compare(neuron.getOutput(), 1));
		neuron.addInputValue(-2.0f);
		TEST_ASSERT(compare(neuron.getOutput(), 1)); 
		neuron.update();
		TEST_ASSERT(compare(neuron.getOutput(), 0));
		neuron.setActivationType(NeuralNet::Activation::Type::tanh_);
		neuron.addInputValue(1.0f);
		neuron.update();
		TEST_ASSERT(compare(neuron.getOutput(), 0));
		neuron.setActivationType(NeuralNet::Activation::Type::sigmoid);
		neuron.update();
		TEST_ASSERT(compare(neuron.getOutput(), 0.5));

		neuron.clearValue();
		neuron.setActivationType(NeuralNet::Activation::Type::linear);
		neuron.addInputValue(1.0f);
		neuron.update();
		TEST_ASSERT(compare(neuron.getOutput(), 1));

		neuron.setInputValues({ 1.0f , 5.f, 3.f});
		neuron.update();
		TEST_ASSERT(compare(neuron.getOutput(), 9.f));*/


		TEST_END;
	}

};
