#include "MainWindow.h"
#include "ui_MainWindow.h"
#include <iostream>
#include <QCloseEvent>
#include <QDebug>

using namespace QSFML;
using namespace QSFML::Objects;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_canvas = nullptr;
    setupCanvas();
    
    
    m_trainingData.push_back({ {0,0, 1},{0} });
    m_trainingData.push_back({ {0,1, 1},{1} });
    m_trainingData.push_back({ {1,0, 1},{1} });
    m_trainingData.push_back({ {1,1, 1},{0} });


    connect(&m_timer, &QTimer::timeout, this, &MainWindow::onTimerFinish);
    m_timer.start(10);
    
}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_canvas;
}

void MainWindow::setupCanvas()
{
    CanvasSettings settings;
    //settings.layout.autoAjustSize = false;
    settings.layout.fixedSize = sf::Vector2u(300, 100);
    settings.contextSettings.antialiasingLevel = 8;
    settings.timing.frameTime = 0;
    //settings.updateControlls.enableMultithreading = false;
    //settings.updateControlls.enablePaintLoop = false;
    //settings.updateControlls.enableEventLoop = false;
    //settings.updateControlls.enableUpdateLoop = false;
    m_canvas = new Canvas(ui->canvasWidget, settings);

    DefaultEditor* defaultEditor = new DefaultEditor();
    defaultEditor->getCamera()->setMinZoom(0.01);
    m_canvas->addObject(defaultEditor);
    qDebug() << defaultEditor->toString().c_str();



    m_net = new NeuralNet::FullConnectedNeuralNet(3, 0, 1, 1);
    m_visuNet = m_net->createVisualisation();
    m_visuNet->setPosition(sf::Vector2f(50, 50));
    std::vector<float> weights = m_net->getWeights();
    for (size_t i = 0; i < weights.size(); i++)
    {
        weights[i] = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
            //(2.f *i / weights.size())-1.f;
	}
    m_net->setWeights(weights);
    m_net->setActivationType(NeuralNet::Activation::Type::gaussian);
    //m_net->setActivationType(1, 0, NeuralNet::Activation::Type::finiteLinear);
    m_canvas->addObject(m_visuNet);


    m_customNet = new NeuralNet::CustomConnectedNeuralNet(3, 1);

    // m_customNet->addConnection(0, 3, 1);
    // m_customNet->addConnection(1, 3, 1);
    // m_customNet->addConnection(2, 3, -1);
    m_customNet->addConnection(0, 3, 1);
    m_customNet->addConnection(0, 4, 1);
    m_customNet->addConnection(1, 3, 1);
    m_customNet->addConnection(2, 3, 1);

    m_customNet->addConnection(2, 4, -1);
    m_customNet->addConnection(1, 4, 1);
   // m_customNet->addConnection(3, 4, 1);
    m_customNet->addConnection(3, 5, 1);
    m_customNet->addConnection(4, 5, 1);
    m_customNet->addConnection(1, 5, 1);
    m_customNet->buildNetwork();
    m_customNet->setActivationType(NeuralNet::Activation::Type::gaussian);
    m_customNet->setActivationType(5, NeuralNet::Activation::Type::tanh_);
    m_customNet->setActivationType(3, NeuralNet::Activation::Type::relu);
    m_customNet->setInputValues({ -1, 0.5, 1 });
    m_customNet->update();

    QSFML::Objects::CanvasObject* customVisuNet = new QSFML::Objects::CanvasObject("CustomConnectedNeuralNet");
    customVisuNet->setPosition(sf::Vector2f(50, 300));
    NeuralNet::Visualisation::CustomConnectedNeuralNetPainter* customPainter = m_customNet->createVisualisation();
    customVisuNet->addComponent((QSFML::Components::Component*)customPainter);
    m_canvas->addObject(customVisuNet);

}
void MainWindow::closeEvent(QCloseEvent* event)
{
    if (m_canvas)
        m_canvas->stop();
    event->accept();
}


void MainWindow::onTimerFinish()
{
    static int currentExampleIndex = 0;

    TrainingSample& trainSample = m_trainingData[currentExampleIndex];
    m_net->setInputValues(trainSample.inputs);
    m_net->update();
    float netError1 = m_net->getNetError(trainSample.expectedOutput);
    m_net->learn(trainSample.expectedOutput);

    m_customNet->setInputValues(trainSample.inputs);
    m_customNet->update();
    float netError2 = m_customNet->getNetError(trainSample.expectedOutput);
    m_customNet->learn(trainSample.expectedOutput);

    std::cout << netError1 << "\t" << netError2 << "\n";

    currentExampleIndex++;
    if (currentExampleIndex >= m_trainingData.size())
        currentExampleIndex = 0;

    static int counter = 0; 
    counter++;
    if (counter == 1000)
    {
        const auto netWeights = m_net->getWeights();
        std::cout << "weights: ";
        for (size_t i = 0; i < netWeights.size(); i++)
        {
			std::cout << netWeights[i] << "\t";
		}
        std::cout << "\n";
        m_timer.setInterval(1000);
    }
}
