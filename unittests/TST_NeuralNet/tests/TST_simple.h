#pragma once

#include "UnitTest.h"
#include <QObject>
#include <QCoreapplication>





class TST_simple : public UnitTest::Test
{
	TEST_CLASS(TST_simple)
public:
	TST_simple()
		: Test("TST_simple")
	{
		ADD_TEST(test1);
		ADD_TEST(test2);

	}

private:

	// Tests
	TEST_FUNCTION(test1)
	{
		TEST_START(results);

		int a = 0;
		TEST_MESSAGE("is a == 0?");
		TEST_ASSERT(a == 0);


	}




	TEST_FUNCTION(test2)
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


	}

};
