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
	m_net = nullptr;
	m_customNet = nullptr;
    m_netObject1 = nullptr;
    m_netObject2 = nullptr;
    ui->setupUi(this);
    m_canvas = nullptr;
    setupCanvas();

    connect(&m_timer, &QTimer::timeout, this, &MainWindow::onTimerFinish);
    m_timer.start(0);

    
    
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
    settings.timing.frameTime = 0.015;
    //settings.updateControlls.enableMultithreading = false;
    //settings.updateControlls.enablePaintLoop = false;
    //settings.updateControlls.enableEventLoop = false;
    //settings.updateControlls.enableUpdateLoop = false;
    m_canvas = new Canvas(ui->canvasWidget, settings);

    DefaultEditor* defaultEditor = new DefaultEditor("Grid", sf::Vector2f(1600, 16000));
    defaultEditor->getCamera()->setMinZoom(0.05);
    m_canvas->addObject(defaultEditor);
    qDebug() << defaultEditor->toString().c_str();


    m_dataset.load("../dataset/train");
    m_validationSet.load("../dataset/validation");


    unsigned int inps = 2;
    unsigned int outps = 1;
    unsigned int layerCount = 1;

    if (m_dataset.getData().size() == 0)
    {
        qDebug() << "Dataset is empty";
        return;
    }
    inps = m_dataset.getInputSize();
    outps = m_dataset.getOutputSize();
    std::vector < NeuralNet::Neuron::ID> inputIDs(inps, 0);
    std::vector < NeuralNet::Neuron::ID> outputIDs(outps, 0);
    for (size_t i = 0; i < inputIDs.size(); ++i)
        inputIDs[i] = i;
    for (size_t i = 0; i < outputIDs.size(); ++i)
        outputIDs[i] = inputIDs.size()+i;
    
    
    m_net = new NeuralNet::FullConnectedNeuralNet(inputIDs, layerCount, 10, outputIDs);

    m_net->enableSoftMaxOutput(true);
    m_net->setActivationType(NeuralNet::Activation::Type::tanh_);
    m_net->setLayerActivationType(layerCount+1, NeuralNet::Activation::Type::tanh_);
    //m_net->setLayerActivationType(2, NeuralNet::Activation::Type::tanh_);
    m_net->setLearningRate(0.1);
    //m_net->enableNormalizedNetInput(true);

    m_netObject1 = new NeuralNet::NeuralNetCanvasObject(m_net, "NeuralNetCanvasObject1");
    m_netObject1->setLayerSpacing(300);
    m_netObject1->setPosition(sf::Vector2f(0, 0));
   // m_netObject1->setNeuronRadius(1);
    m_netObject1->resetPositions();
    m_netObject1->enableNeuronGraphOfLayer(m_netObject1->getInputLayerIndex(), false);
    m_netObject1->enableNeuronTextOfLayer(m_netObject1->getInputLayerIndex(), false);

    // Set neuron positions
    sf::Vector2u dim = m_dataset.getDimensions();
    sf::Vector2f spacing(20, 20);
    for (unsigned int x = 0; x < dim.x; ++x)
    {
        float xPos = x * spacing.x;
        for (unsigned int y = 0; y < dim.y; ++y)
        {
			float yPos = y * spacing.y;
			m_netObject1->setNeuronPosition(x * dim.y + y, sf::Vector2f(xPos, yPos));
        }
    }
    float layerSpacing = 300;
    for (unsigned int i = 0; i < layerCount+1; ++i)
    {
        m_netObject1->resetLayerPosition(i+1, sf::Vector2f(spacing.x* dim.x+50 + i* layerSpacing, 0), sf::Vector2f(0, 30));
    }
    
    m_canvas->addObject(m_netObject1);
    m_canvas->applyObjectChanges();

    std::cout << "Objects: \n" << m_canvas->getObjectsTreeString() << "\n";
}
void MainWindow::closeEvent(QCloseEvent* event)
{
    if (m_canvas)
        m_canvas->stop();
    event->accept();
}

void MainWindow::onTimerFinish()
{
    train(100);
}

void MainWindow::on_pauseTraining_pushButton_clicked()
{
    static bool paused = false;
    paused = !paused;
    if(paused)
		m_timer.stop();
	else
		m_timer.start();
}
void MainWindow::on_testNext_pushButton_clicked()
{
    const std::vector<Dataset::DataPoint>& dataset = m_validationSet.getData();
    const Dataset::DataPoint& dataPoint = dataset[rand() % dataset.size()];
    float error = test(dataPoint);
    std::cout << "Error: " << error << "\n";
}
void MainWindow::on_reset_pushButton_clicked()
{
    auto w = m_net->getWeights();
    for (size_t i = 0; i < w.size(); ++i)
    {
        w[i] = (float)(rand() % 2000) / 1000 - 1.0f;
    }
    m_net->setWeights(w);
}


void MainWindow::train(size_t iterations)
{
    if (iterations == 0)
        return;
    const std::vector<Dataset::DataPoint> &dataset = m_dataset.getData();
    float error = 0;
    std::unordered_map<NeuralNet::Neuron::ID, NeuralNet::Neuron*> neurons = m_net->getNeurons();
    struct ChangeTrack
    {
        float lastVal = 0;
        float maxChange = 0;
    };
    std::unordered_map<NeuralNet::Neuron::ID, ChangeTrack> maxChangeList;
    const bool checkUnchangedNeurons = true;

    static std::vector<float> digitErrors;
    if (digitErrors.size() != m_net->getOutputCount())
        digitErrors = std::vector<float>(m_net->getOutputCount(), 0);
    for (size_t i = 0; i < iterations; i++)
	{
		//for (const Dataset::DataPoint& dataPoint : dataset)
        const Dataset::DataPoint& dataPoint = dataset[rand() % dataset.size()];
		m_net->setInputValues(dataPoint.features);
		m_net->update();
		m_net->learn(dataPoint.labels);
        float netError = m_net->getNetError(dataPoint.labels);
        error += netError;
        for(size_t j=0; j<dataPoint.labels.size(); ++j)
            if (dataPoint.labels[j] > 0.9)
            {
                digitErrors[j] = digitErrors[j] * 0.99f + netError * 0.01f;
                break;
            }
        

        if (checkUnchangedNeurons)
        {
            for (auto& neuron : neurons)
            {
                if (neuron.second->getInputConnections().size() == 0)
                    continue;
                float neuronOutput = neuron.second->getOutput();
                if (maxChangeList.find(neuron.second->getID()) == maxChangeList.end())
                {
                    maxChangeList[neuron.second->getID()].lastVal = neuronOutput;
                    maxChangeList[neuron.second->getID()].maxChange = 0;
                    continue;
                }
                ChangeTrack current = maxChangeList[neuron.second->getID()];
                float diff = std::abs(neuronOutput - current.lastVal);
                maxChangeList[neuron.second->getID()].lastVal = neuronOutput;
                if (diff > current.maxChange)
                    maxChangeList[neuron.second->getID()].maxChange = diff;
            }
        }
	}
    if (checkUnchangedNeurons)
    {
        for (auto& neuron : maxChangeList)
        {
            float diff = neuron.second.maxChange;
            if (diff < 0.3)
            {
                const auto& inputs = m_net->getNeuron(neuron.first)->getInputConnections();
                for (auto& inp : inputs)
                    inp->setWeight((float)(rand() % 2000) / 1000 - 1.f);
            }
        }
    }

	error /= iterations;
    static float averageError = 0;
    averageError = 0.99f * averageError + 0.01f * error;
	std::cout << "Error: " << averageError << "\t";
    for (size_t j = 0; j < digitErrors.size(); ++j)
    {
        std::cout << "[" << j << ":" << digitErrors[j] << "] ";
    }
    std::cout << "\n";
}
float MainWindow::test(const Dataset::DataPoint& dataPoint)
{
    m_net->setInputValues(dataPoint.features);
    m_net->update();
    return m_net->getNetError(dataPoint.labels);
}