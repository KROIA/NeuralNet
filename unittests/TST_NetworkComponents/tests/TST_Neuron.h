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

		NeuralNet::Neuron neuron;

		TEST_ASSERT(compare(neuron.getOutput(), 0), "Output shuld be 0");
		neuron.update();
		TEST_ASSERT(compare(neuron.getOutput(), 0.5), "Output shuld be 0.5");

		neuron.addInputValue(1.0f);
		neuron.setActivationType(NeuralNet::Activation::Type::linear);
		neuron.update();

		float outp = neuron.getOutput();
		TEST_ASSERT(compare(outp, 1), "Output shuld be 1");

		TEST_END;
	}




	bool calcOutput(TestResults& results)
	{
		TEST_START(results);

		int a = 0;
		TEST_ASSERT_M(a == 0, "is a == 0?");

		int b = 0;
		if (b != 0)
		{
			TEST_FAIL("b is not 0");
		}

		// fails if a != b
		TEST_COMPARE(a, b);

		TEST_END;
	}

};
