#ifdef QT_ENABLED
#include <QCoreApplication>
#endif
#include <iostream>
#include "NeuralNet.h"
#include <iostream>
#include "tests.h"

#ifdef QT_WIDGETS_ENABLED
#include <QWidget>
#include <QApplication>
#endif

// Instantiate Tests here:
// TEST_INSTANTIATE(Test_simple); // Where Test_simple is a derived class from the Test class
TEST_INSTANTIATE(TST_Activation);
TEST_INSTANTIATE(TST_Neuron);
TEST_INSTANTIATE(TST_Connection);
TEST_INSTANTIATE(TST_FullConnectedNeuralNet);
TEST_INSTANTIATE(TST_BackpropagationXOR);
TEST_INSTANTIATE(TST_GeneticLearnXOR);


int main(int argc, char* argv[])
{
#ifdef QT_WIDGETS_ENABLED
	QGuiApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
	QGuiApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
	QGuiApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
#endif
#ifdef QT_ENABLED
	#ifdef QT_WIDGETS_ENABLED
		QApplication app(argc, argv);
	#else
		QCoreApplication app(argc, argv);
	#endif
#endif

	NeuralNet::LibraryInfo::printInfo();

	std::cout << "Running "<< UnitTest::Test::getTests().size() << " tests...\n";
	UnitTest::Test::TestResults results;
	UnitTest::Test::runAllTests(results);
	UnitTest::Test::printResults(results);

#ifdef QT_WIDGETS_ENABLED
	//QWidget* widget = NeuralNet::LibraryInfo::createInfoWidget();
	//if (widget)
	//	widget->show();
#endif
#ifdef QT_ENABLED
	return app.exec();
#else
	return 0;
#endif
}
