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
    settings.timing.frameTime = 0;
    //settings.updateControlls.enableMultithreading = false;
    //settings.updateControlls.enablePaintLoop = false;
    //settings.updateControlls.enableEventLoop = false;
    //settings.updateControlls.enableUpdateLoop = false;
    m_canvas = new Canvas(ui->canvasWidget, settings);

    DefaultEditor* defaultEditor = new DefaultEditor("Grid", sf::Vector2f(1600, 16000));
    defaultEditor->getCamera()->setMinZoom(0.05);
    m_canvas->addObject(defaultEditor);
    qDebug() << defaultEditor->toString().c_str();


    m_dataset.load("dataset/train");
    m_validationSet.load("dataset/validation");


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
    
    
    m_net = new NeuralNet::FullConnectedNeuralNet(inps, layerCount,15, outps);

    m_net->setActivationType(NeuralNet::Activation::Type::tanh_);
    m_net->setLayerActivationType(layerCount+1, NeuralNet::Activation::Type::tanh_);
    //m_net->setLayerActivationType(2, NeuralNet::Activation::Type::tanh_);
    m_net->setLearningRate(0.01);
    //m_net->enableNormalizedNetInput(true);

    m_netObject1 = new NeuralNet::NeuralNetCanvasObject(m_net, "NeuralNetCanvasObject1");
    m_netObject1->setLayerSpacing(300);
    m_netObject1->setPosition(sf::Vector2f(0, 0));
   // m_netObject1->setNeuronRadius(1);
    m_netObject1->resetPositions();

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


void MainWindow::train(size_t iterations)
{
    if (iterations == 0)
        return;
    const std::vector<Dataset::DataPoint> &dataset = m_dataset.getData();
    float error = 0;
    for (size_t i = 0; i < iterations; i++)
	{
		//for (const Dataset::DataPoint& dataPoint : dataset)
        const Dataset::DataPoint& dataPoint = dataset[rand() % dataset.size()];
		m_net->setInputValues(dataPoint.features);
		m_net->update();
		m_net->learn(dataPoint.labels);
        error += m_net->getNetError(dataPoint.labels);
	}
	error /= iterations;
    static float averageError = 0;
    averageError = 0.99f * averageError + 0.01f * error;
	std::cout << "Error: " << averageError << "\n";
}
float MainWindow::test(const Dataset::DataPoint& dataPoint)
{
    m_net->setInputValues(dataPoint.features);
    m_net->update();
    return m_net->getNetError(dataPoint.labels);
}