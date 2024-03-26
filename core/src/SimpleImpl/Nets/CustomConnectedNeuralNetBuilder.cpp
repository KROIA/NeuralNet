#include "SimpleImpl/Nets/CustomConnectedNeuralNet.h"
#include <queue>
#include "SimpleImpl/NetworkComponents/InputNeuron.h"

namespace NeuralNet
{
	void CustomConnectedNeuralNet::CustomConnectedNeuralNetBuilder::buildNetwork(
		const std::vector<ConnectionInfo>& connections,
		unsigned int inputCount, 
		unsigned int outputCount,
		NetworkData &network)
	{
		// Clear old network
		destroyNetwork(network);

		std::unordered_map<NeuronID, Neuron*> &neurons = network.neurons;
		std::vector<Connection*> &connectionsOut = network.connections;
//		std::vector<Layer> &layers = network.layers;

		//std::vector<std::vector<ConnectionInfo>> layeredConnections;
		//splitIntoLayers(connections, layeredConnections);
		
		// Check which neuron IDs are needed and create them
		/*layers.resize(layeredConnections.size());
		for (size_t i = 0; i < layeredConnections.size(); ++i)
		{
			std::vector<ConnectionInfo> &layerConnections = layeredConnections[i];
			Layer &layer = layers[i];
			std::vector<Neuron*> &layerNeurons = layer.neurons;
			std::vector<Connection*> &layerInputConnections = layer.inputConnections;

			layerNeurons.reserve(layerConnections.size());
			layerInputConnections.reserve(layerConnections.size());
			for (auto& connection : layerConnections)
			{
				if (i == 0)
				{
					if (neurons.find(connection.fromNeuronID) == neurons.end())
					{
						InputNeuron *n = new InputNeuron();
						neurons[connection.fromNeuronID] = n;
						layerNeurons.push_back(n);
					}
					if (neurons.find(connection.toNeuronID) == neurons.end())
					{
						InputNeuron* n = new InputNeuron();
						neurons[connection.toNeuronID] = new InputNeuron();
						layerNeurons.push_back(n);
					}
						
				}
				else
				{
					if (neurons.find(connection.fromNeuronID) == neurons.end())
						neurons[connection.fromNeuronID] = new Neuron();
					if (neurons.find(connection.toNeuronID) == neurons.end())
						neurons[connection.toNeuronID] = new Neuron();
				}
				
			}
			
		}*/



		std::unordered_map<NeuronID, bool> ids;
		ids.reserve(connections.size());
		for (auto& connection : connections)
		{
			if (ids.find(connection.fromNeuronID) == ids.end())
			{
				ids[connection.fromNeuronID] = true;
			}
			if (ids.find(connection.toNeuronID) == ids.end())
			{
				ids[connection.toNeuronID] = true;
			}
		}
		std::vector<NeuronID> sortedIds;
		sortedIds.reserve(ids.size());
		for (auto& id : ids)
		{
			sortedIds.push_back(id.first);
		}
		std::sort(sortedIds.begin(), sortedIds.end());

		std::unordered_map<NeuronID, bool> inputNeurons;
		std::unordered_map<NeuronID, bool> outputNeurons;

		// get the first N neuron IDs as input neurons
		for (unsigned int i = 0; i < inputCount; ++i)
		{
			inputNeurons[sortedIds[i]] = true;
		}

		// get the last N neuron IDs as output neurons
		for (unsigned int i = 0; i < outputCount; ++i)
		{
			outputNeurons[sortedIds[sortedIds.size() - 1 - i]] = true;
		}



		// Create neurons
		for (auto& connection : connections)
		{
			if (neurons.find(connection.fromNeuronID) == neurons.end())
			{
				if (inputNeurons.find(connection.fromNeuronID) != inputNeurons.end())
				{
					neurons[connection.fromNeuronID] = new InputNeuron();
				}
				else
				{
					neurons[connection.fromNeuronID] = new Neuron();
				}
			}
			if(neurons.find(connection.toNeuronID) == neurons.end())
				neurons[connection.toNeuronID] = new Neuron();
		}

		// Create connections
		connectionsOut.reserve(connections.size());
		for (auto& connection : connections)
		{
			Connection* conn = new Connection(neurons[connection.fromNeuronID], neurons[connection.toNeuronID], connection.weight);
			connectionsOut.push_back(conn);
		}

		splitIntoLayers(network);


		Layer lastLayer = network.layers[network.layers.size() - 1];
		if (lastLayer.neurons.size() != outputNeurons.size())
		{

			Layer newSecondlastLayer;
			Layer newOutputLayer;
			for (auto& it : outputNeurons)
			{
				NeuronID id = it.first;
				for (int i = 0; i < lastLayer.neurons.size(); ++i)
				{
					if (lastLayer.neurons[i] == network.neurons[id])
					{
						newOutputLayer.neurons.push_back(lastLayer.neurons[i]);
						for (auto& conn : lastLayer.inputConnections)
						{
							if (conn->getEndNeuron() == lastLayer.neurons[i])
							{
								newOutputLayer.inputConnections.push_back(conn);
							}
						}
					}
					else
					{
						newSecondlastLayer.neurons.push_back(lastLayer.neurons[i]);
						for (auto& conn : lastLayer.inputConnections)
						{
							if (conn->getEndNeuron() == lastLayer.neurons[i])
							{
								newSecondlastLayer.inputConnections.push_back(conn);
							}
						}
					}
				}
			}
			network.layers.pop_back();
			network.layers.push_back(newSecondlastLayer);
			network.layers.push_back(newOutputLayer);
		}
	}

	void CustomConnectedNeuralNet::CustomConnectedNeuralNetBuilder::destroyNetwork(
		NetworkData& network)
	{
		network.layers.clear();

		for(auto& neuron : network.neurons)
			delete neuron.second;
		network.neurons.clear();

		for(auto& conn : network.connections)
			delete conn;
		network.connections.clear();
	}


	void CustomConnectedNeuralNet::CustomConnectedNeuralNetBuilder::getConnections(
		const NetworkData& network,
		std::vector<ConnectionInfo>& connectionInfos)
	{
		std::unordered_map<Neuron*, NeuronID> neuronIDs;
		for (auto& neuron : network.neurons)
			neuronIDs[neuron.second] = neuron.first;

		connectionInfos.clear();
		connectionInfos.reserve(network.connections.size());
		for (auto& connection : network.connections)
		{
			ConnectionInfo info;
			NeuronID fromID, toID;

			auto fromIDIt = neuronIDs.find(connection->getStartNeuron());
			auto toIDIt = neuronIDs.find(connection->getEndNeuron());
			if (fromIDIt == neuronIDs.end() || toIDIt == neuronIDs.end())
				continue;
			info.fromNeuronID = fromIDIt->second;
			info.toNeuronID = toIDIt->second;
			info.weight = connection->getWeight();
			connectionInfos.push_back(info);
		}
	}


	void CustomConnectedNeuralNet::CustomConnectedNeuralNetBuilder::splitIntoLayers(NetworkData& network)
	{
		// Clear any existing layers
		network.layers.clear();

		struct NeuronInfo
		{
			NeuronID id;
			bool visited = false;
		};
		std::unordered_map<Neuron*, NeuronInfo> neuronIDs;
		for (auto& neuron : network.neurons)
			neuronIDs[neuron.second].id = neuron.first;



		/* // Clear any existing layers
		network.layers.clear();

		// Initialize a queue for BFS
		std::queue<Neuron*> neuronQueue;

		std::unordered_map<Neuron*, NeuronID> neuronIDs;
		for (auto& neuron : network.neurons)
			neuronIDs[neuron.second] = neuron.first;

		// Mark all neurons as unvisited
		std::unordered_map<NeuronID, bool> visited;
		for (const auto& neuronPair : network.neurons) {
			visited[neuronPair.first] = false;
		}

		// Start BFS from input neurons
		for (const auto& neuronPair : network.neurons) {
			Neuron* neuron = neuronPair.second;
			bool hasInput = false;
			for (const auto& connection : network.connections) {
				if (connection->getEndNeuron() == neuron) {
					hasInput = true;
					break;
				}
			}
			if (!hasInput) {
				neuronQueue.push(neuron);
				visited[neuronPair.first] = true;
			}
		}

		// Map to store the layer index for each neuron
		std::unordered_map<NeuronID, size_t> neuronLayerMap;

		// Perform BFS to split neurons into layers
		size_t currentLayerIndex = 0;
		while (!neuronQueue.empty()) {
			Layer layer;
			size_t currentLayerSize = neuronQueue.size();
			for (size_t i = 0; i < currentLayerSize; ++i) {
				Neuron* currentNeuron = neuronQueue.front();
				neuronQueue.pop();
				layer.neurons.push_back(currentNeuron);
				neuronLayerMap[neuronIDs[currentNeuron]] = currentLayerIndex;

				// Add input connections to the current layer
				for (Connection* connection : network.connections) {
					if (connection->getEndNeuron() == currentNeuron) {
						if (neuronLayerMap[neuronIDs[connection->getStartNeuron()]] < currentLayerIndex) {
							// Input neuron belongs to a previous layer, add connection to current layer
							layer.inputConnections.push_back(connection);
						}
					}
					if (connection->getStartNeuron() == currentNeuron) {
						Neuron* inputNeuron = connection->getEndNeuron();
						// Add input neuron to queue if not visited yet
						if (!visited[neuronIDs[inputNeuron]]) {
							neuronQueue.push(inputNeuron);
							visited[neuronIDs[inputNeuron]] = true;
						}
					}
				}
			}
			network.layers.push_back(layer);
			++currentLayerIndex;
		}*/
		/*// Clear any existing layers
		network.layers.clear();

		// Initialize a queue for BFS
		std::queue<Neuron*> neuronQueue;

		std::unordered_map<Neuron*, NeuronID> neuronIDs;
		for (auto& neuron : network.neurons)
			neuronIDs[neuron.second] = neuron.first;

		// Mark all neurons as unvisited
		std::unordered_map<NeuronID, bool> visited;
		for (const auto& neuronPair : network.neurons) 
		{
			visited[neuronPair.first] = false;
		}

		// Start BFS from input neurons
		for (const auto& neuronPair : network.neurons)
		{
			Neuron* neuron = neuronPair.second;
			bool hasInput = false;
			for (const auto& connection : network.connections) 
			{
				if (connection->getEndNeuron() == neuron) 
				{
					hasInput = true;
					break;
				}
			}
			if (!hasInput) {
				neuronQueue.push(neuron);
				visited[neuronPair.first] = true;
			}
		}

		// Perform BFS to split neurons into layers
		while (!neuronQueue.empty()) 
		{
			Layer layer;
			size_t currentLayerSize = neuronQueue.size();
			std::queue<Neuron*> tempQueue = neuronQueue;
			neuronQueue = std::queue<Neuron*>();
			for (size_t i = 0; i < currentLayerSize; ++i) 
			{
				Neuron* currentNeuron = tempQueue.front();
				tempQueue.pop();
				layer.neurons.push_back(currentNeuron);

				// Add input connections to the current layer
				for (Connection* connection : network.connections) 
				{
					if (connection->getEndNeuron() == currentNeuron) 
					{
						layer.inputConnections.push_back(connection);
					}
					if (connection->getStartNeuron() == currentNeuron)
					{
						Neuron* inputNeuron = connection->getEndNeuron();
						// Add input neuron to queue if not visited yet
						if (!visited[neuronIDs[inputNeuron]])
						{
							neuronQueue.push(inputNeuron);
							visited[neuronIDs[inputNeuron]] = true;
						}
					}
				}
			}
			network.layers.push_back(layer);
		}*/
	}

	/*
	void CustomConnectedNeuralNet::CustomConnectedNeuralNetBuilder::splitIntoLayers(
		const std::vector<ConnectionInfo>& connections, 
		std::vector<std::vector<ConnectionInfo>>& layers)
	{
		// Step 2: Create adjacency list
		std::unordered_map<unsigned int, std::vector<unsigned int>> adjacencyList;
		for (const auto& conn : connections) {
			adjacencyList[conn.fromNeuronID].push_back(conn.toNeuronID);
		}

		// Step 3: Perform BFS traversal to sort neurons into layers
		layers.clear();
		layers.push_back(std::vector<ConnectionInfo>()); // Start with the first layer
		BFS(adjacencyList, layers);
	}

	// Helper function to perform BFS traversal
	void CustomConnectedNeuralNet::CustomConnectedNeuralNetBuilder::BFS(
		const std::unordered_map<unsigned int, std::vector<unsigned int>>& adjacencyList, 
		std::vector<std::vector<ConnectionInfo>>& layers)
	{
		std::queue<unsigned int> q;
		std::unordered_map<unsigned int, bool> visited;

		for (const auto& entry : adjacencyList) {
			if (entry.second.empty()) {
				q.push(entry.first);
				visited[entry.first] = true;
			}
		}

		while (!q.empty()) {
			unsigned int currentNeuronID = q.front();
			q.pop();

			// Add neuron to the current layer
			layers.back().push_back({ 0, currentNeuronID, 0.0f });

			// Visit all neighbors of current neuron
			for (unsigned int neighbor : adjacencyList.at(currentNeuronID)) {
				if (!visited[neighbor]) {
					visited[neighbor] = true;
					q.push(neighbor);
				}
			}

			// If queue is empty and not all neurons are visited, move to the next layer
			if (q.empty() && visited.size() < adjacencyList.size()) {
				layers.push_back(std::vector<ConnectionInfo>());
				for (const auto& entry : adjacencyList) {
					if (!visited[entry.first]) {
						q.push(entry.first);
						visited[entry.first] = true;
						break;
					}
				}
			}
		}
	}*/
}