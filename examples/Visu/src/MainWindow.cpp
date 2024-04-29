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
    
    
    m_trainingData.push_back({ {0,0},{0} });
    m_trainingData.push_back({ {0,1},{1} });
    m_trainingData.push_back({ {1,0},{1} });
    m_trainingData.push_back({ {1,1},{0} });


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


    unsigned int inps = 2;
    unsigned int outps = 1;
    if (m_trainingData.size() > 0)
    {
        inps = m_trainingData[0].inputs.size();
        outps = m_trainingData[0].expectedOutput.size();
    }
    std::vector < NeuralNet::Neuron::ID> inputIDs(inps, 0);
    std::vector < NeuralNet::Neuron::ID> outputIDs(outps, 0);
    for (size_t i = 0; i < inputIDs.size(); ++i)
        inputIDs[i] = i;
    for (size_t i = 0; i < outputIDs.size(); ++i)
        outputIDs[i] = inputIDs.size() + i;

    m_net = new NeuralNet::FullConnectedNeuralNet(inputIDs, 1, 2, outputIDs);
    //QSFML::Objects::CanvasObject* customVisuNet1 = new QSFML::Objects::CanvasObject("CustomConnectedNeuralNet");
    //customVisuNet1->setPosition(sf::Vector2f(50, 50));
    //NeuralNet::Visualisation::CustomConnectedNeuralNetPainter* visu1  = m_net->createVisualisation();
    //customVisuNet1->addComponent(visu1);
   /* std::vector<float> weights = m_net->getWeights();
    for (size_t i = 0; i < weights.size(); i++)
    {
        weights[i] = QSFML::Utilities::RandomEngine::getFloat(-1, 1);
            //(2.f *i / weights.size())-1.f;
	}
    m_net->setWeights(weights);*/
    m_net->setActivationType(NeuralNet::Activation::Type::sigmoid);
    m_net->setActivationType(outputIDs[0], NeuralNet::Activation::Type::gaussian);
    //m_canvas->addObject(customVisuNet1);

    m_netObject1 = new NeuralNet::NeuralNetCanvasObject(m_net, "NeuralNetCanvasObject1");
    m_netObject1->setPosition(sf::Vector2f(50, 50));
    m_canvas->addObject(m_netObject1);


    
    m_customNet = new NeuralNet::CustomConnectedNeuralNet(inputIDs, outputIDs);


    /*for (int i = 3; i < 8; ++i)
    {
        m_customNet->addConnection(0, i, QSFML::Utilities::RandomEngine::getFloat(-1,1));
        m_customNet->addConnection(1, i, QSFML::Utilities::RandomEngine::getFloat(-1,1));
        m_customNet->addConnection(2, i, QSFML::Utilities::RandomEngine::getFloat(-1,1));

        m_customNet->addConnection(i, 20, QSFML::Utilities::RandomEngine::getFloat(-1,1));
    }*/

    // m_customNet->addConnection(0, 3, 1);
    // m_customNet->addConnection(1, 3, 1);
    // m_cuomNet->addConnection(2, 3, -1);
    m_customNet->addConnection(0, 3, -0.5);
    //m_customNet->addConnection(0, 4);
    m_customNet->addConnection(1, 3, -2);
   // m_customNet->addConnection(2, 3, 0.1);
    m_customNet->setBias(3, -1);

    //m_customNet->addConnection(2, 4);
    //m_customNet->addConnection(1, 4);

   // m_customNet->addConnection(3, 5);
   // m_customNet->addConnection(4, 5);
   // m_customNet->addConnection(1, 5);
    
    m_customNet->setActivationType(NeuralNet::Activation::Type::tanh_);
    m_customNet->setActivationType(3, NeuralNet::Activation::Type::gaussian);
    m_customNet->buildNetwork();
    
   /* m_customNet->setActivationType(3, NeuralNet::Activation::Type::sigmoid);
    m_customNet->setActivationType(4, NeuralNet::Activation::Type::relu);
    m_customNet->setActivationType(5, NeuralNet::Activation::Type::finiteLinear);
    m_customNet->setActivationType(6, NeuralNet::Activation::Type::gaussian);
    m_customNet->setActivationType(7, NeuralNet::Activation::Type::binary);*/
//    m_customNet->setInputValues({ -1, 0.5, 1 });
//    m_customNet->update();

   //QSFML::Objects::CanvasObject* customVisuNet = new QSFML::Objects::CanvasObject("CustomConnectedNeuralNet");
   //customVisuNet->setPosition(sf::Vector2f(50, 300));
   //NeuralNet::Visualisation::CustomConnectedNeuralNetPainter* customPainter = m_customNet->createVisualisation();
   //customVisuNet->addComponent((QSFML::Components::Component*)customPainter);
   //m_canvas->addObject(customVisuNet);


    m_netObject2 = new NeuralNet::NeuralNetCanvasObject(m_customNet, "NeuralNetCanvasObject2");
    m_netObject2->setPosition(sf::Vector2f(50, 300));
    m_canvas->addObject(m_netObject2);

}
void MainWindow::closeEvent(QCloseEvent* event)
{
    if (m_canvas)
        m_canvas->stop();
    event->accept();
}

void shrinkNetwork(std::vector<NeuralNet::ConnectionInfo>& connections)
{
    float weightThreshold = 0.3f;
    std::vector<NeuralNet::ConnectionInfo> newConnections;
    std::unordered_map < NeuralNet::Neuron::ID, int> outputCount;
    std::unordered_map < NeuralNet::Neuron::ID, int> inputCount;

    newConnections.reserve(connections.size());

    for (const auto &connection : connections)
    {
        float weight = std::abs(connection.weight);
        if (weight >= weightThreshold)
        {
			newConnections.push_back(connection);
            outputCount[connection.fromNeuronID]++;
            inputCount[connection.toNeuronID]++;
		}
	}

    for (int i = 0; i < newConnections.size(); ++i)
    {
        if (outputCount[newConnections[i].toNeuronID] == 0/* ||
            inputCount[newConnections[i].fromNeuronID] == 0*/)
        {
			newConnections.erase(newConnections.begin() + i);
			i--;
		}
    }

    connections = newConnections;
}

void MainWindow::onTimerFinish()
{
    static int currentExampleIndex = 0;
    if (currentExampleIndex >= m_trainingData.size())
    {
       /* std::vector<NeuralNet::ConnectionInfo> connections = m_customNet->getConnections();
        shrinkNetwork(connections);
        if (connections.size() == 0)
        {
            connections = m_customNet->getConnections();
        }
        m_customNet->setConnections(connections);
        m_customNet->buildNetwork();*/

        currentExampleIndex = 0;
    }

    TrainingSample& trainSample = m_trainingData[currentExampleIndex];
    m_net->setInputValues(trainSample.inputs);
    m_net->update();
    float netError1 = m_net->getNetError(trainSample.expectedOutput);
    m_net->learn(trainSample.expectedOutput);

    m_customNet->setInputValues(trainSample.inputs);
    m_customNet->update();
    float netError2 = m_customNet->getNetError(trainSample.expectedOutput);
    m_customNet->learn(trainSample.expectedOutput);

   // std::cout << netError1 << "\t" << netError2 << "\n";

    currentExampleIndex++;
    

    static int counter = 0; 
    counter++;
    if (counter == 1000)
    {
        const auto netWeights = m_net->getWeights();
       /* std::cout << "weights: ";
        for (size_t i = 0; i < netWeights.size(); i++)
        {
			std::cout << netWeights[i] << "\t";
		}
        std::cout << "\n";*/
        m_timer.setInterval(1000);
    }
}
