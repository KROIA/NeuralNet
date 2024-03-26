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

		NeuralNet::InputNeuron neuron1;
		NeuralNet::Neuron neuron2;

		NeuralNet::Connection connection(&neuron1, &neuron2, 0.5f);
		
		neuron1.setActivationType(NeuralNet::Activation::Type::linear);
		neuron2.setActivationType(NeuralNet::Activation::Type::linear);
		neuron1.setValue(1.0f);

		// Pass the Value 
		neuron1.update();
		neuron2.update();
		TEST_ASSERT(compare(connection.getOutputValue(), 0.5));
		TEST_ASSERT(compare(neuron2.getOutput(), 0.5));

		neuron1.setValue(2.0f);
		neuron1.update();
		neuron2.update();
		TEST_ASSERT(compare(connection.getOutputValue(), 1));
		TEST_ASSERT(compare(neuron2.getOutput(), 1));

		TEST_END;
	}
};
