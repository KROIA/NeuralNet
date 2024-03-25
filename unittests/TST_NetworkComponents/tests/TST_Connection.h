#pragma once

#include "UnitTest.h"
#include <QObject>
#include <QCoreapplication>
#include "NeuralNet.h"




class TST_Connection : public UnitTest::Test
{
	TEST_CLASS(TST_Connection)
public:
	TST_Connection()
		: Test("TST_Connection")
	{
		ADD_TEST(TST_Connection::connection);

	}

private:
	float m_tollerance = 0.0001f;
	bool compare(float a, float b)
	{
		return abs(a - b) < m_tollerance;
	}

	// Tests
	bool connection(TestResults& results)
	{
		TEST_START(results);

		NeuralNet::Neuron neuron1;
		NeuralNet::Neuron neuron2;

		NeuralNet::Connection connection(&neuron1, &neuron2, 0.5f);
		
		neuron1.setActivationType(NeuralNet::Activation::Type::linear);
		neuron2.setActivationType(NeuralNet::Activation::Type::linear);
		neuron1.addInputValue(1.0f);
		TEST_ASSERT(compare(connection.getOutputValue(), 0));
		TEST_ASSERT(compare(neuron2.getOutput(), 0));
		neuron2.update();
		TEST_ASSERT(compare(neuron2.getOutput(), 0));
		neuron1.update();
		neuron2.update();
		TEST_ASSERT(compare(connection.getOutputValue(), 0.5));
		TEST_ASSERT(compare(neuron2.getOutput(), 0));

		// Pass the Value 
		connection.passValue();
		neuron2.update();
		TEST_ASSERT(compare(connection.getOutputValue(), 0.5));
		TEST_ASSERT(compare(neuron2.getOutput(), 0.5));

		neuron1.clearValue();
		neuron2.clearValue();
		neuron1.addInputValue(2.0f);
		neuron1.update();
		connection.passValue();
		neuron2.update();
		TEST_ASSERT(compare(connection.getOutputValue(), 1));
		TEST_ASSERT(compare(neuron2.getOutput(), 1));

		TEST_END;
	}
};
