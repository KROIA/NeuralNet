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

		std::vector<ConnectionInfo> uniqueConnections;
		removeDuplicateConnections(connections, uniqueConnections);

		std::unordered_map<Neuron::ID, Neuron*> &neurons = network.neurons;
		std::vector<Connection*> &connectionsOut = network.connections;



		std::unordered_map<Neuron::ID, bool> ids;
		ids.reserve(uniqueConnections.size());
		for (auto& connection : uniqueConnections)
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
		std::vector<Neuron::ID> sortedIds;
		sortedIds.reserve(ids.size());
		for (auto& id : ids)
		{
			sortedIds.push_back(id.first);
		}
		std::sort(sortedIds.begin(), sortedIds.end());

		std::unordered_map<Neuron::ID, bool> inputNeurons;
		std::unordered_map<Neuron::ID, bool> outputNeurons;

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

		for (auto& connection : uniqueConnections)
		{
			if (neurons.find(connection.fromNeuronID) == neurons.end())
			{
				if (inputNeurons.find(connection.fromNeuronID) != inputNeurons.end())
				{
					neurons[connection.fromNeuronID] = new InputNeuron(connection.fromNeuronID);
				}
				else
				{
					neurons[connection.fromNeuronID] = new Neuron(connection.fromNeuronID);
				}
			}
			if(neurons.find(connection.toNeuronID) == neurons.end())
				neurons[connection.toNeuronID] = new Neuron(connection.toNeuronID);
		}

		// Create connections
		connectionsOut.reserve(uniqueConnections.size());
		for (auto& connection : uniqueConnections)
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
				Neuron::ID id = it.first;
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
	
		sortLayers(network);
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
		std::unordered_map<Neuron*, Neuron::ID> neuronIDs;
		for (auto& neuron : network.neurons)
			neuronIDs[neuron.second] = neuron.first;

		connectionInfos.clear();
		connectionInfos.reserve(network.connections.size());
		for (auto& connection : network.connections)
		{
			ConnectionInfo info;
			Neuron::ID fromID, toID;

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
			Neuron::ID id;
			bool visited = false;
		};
		size_t visitedCount = 0;
		std::unordered_map<Neuron*, NeuronInfo> neuronIDs;
		for (auto& neuron : network.neurons)
			neuronIDs[neuron.second].id = neuron.first;

		std::unordered_map<Neuron*, Neuron*> potentialNextLayer;

		// Search input neurons
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
				potentialNextLayer[neuron] = neuron;
				//neuronIDs[neuron].visited = true;
			}
		}

		while (visitedCount < neuronIDs.size())
		{
			std::unordered_map<Neuron*, Neuron*> potentialCurrentLayer = potentialNextLayer;
			std::unordered_map<Neuron*, Neuron*> visitedNeurons;
			//potentialNextLayer.clear();
			Layer layer;

			for (const auto &currentNeuronIt : potentialCurrentLayer)
			{
				bool allInputsVisited = true;
				Neuron* currentNeuron = currentNeuronIt.first;
				const std::vector<Connection*> &inputConnections = currentNeuron->getInputConnections();
				for (const auto& connection : inputConnections)
				{
					Neuron* inputNeuron = connection->getStartNeuron();
					NeuronInfo &info = neuronIDs[inputNeuron];
					if (!info.visited)
					{
						allInputsVisited = false;
						break;
					}
				}

				if (allInputsVisited)
				{
					visitedNeurons[currentNeuron] = currentNeuron;

					layer.neurons.push_back(currentNeuron);
					layer.inputConnections.insert(layer.inputConnections.end(), inputConnections.begin(), inputConnections.end());
				
					// Search for petential next layer neurons
					for (const auto& connection : network.connections)
					{
						if (connection->getStartNeuron() == currentNeuron)
						{
							Neuron* nextNeuron = connection->getEndNeuron();
							if (!neuronIDs[nextNeuron].visited)
							{
								if(potentialNextLayer.find(nextNeuron) == potentialNextLayer.end())
									potentialNextLayer[nextNeuron] = nextNeuron;
							}
						}
					}
				}
			}

			network.layers.push_back(layer);

			visitedCount += visitedNeurons.size();
			for (auto& neuron : visitedNeurons)
			{
				neuronIDs[neuron.first].visited = true;
				potentialNextLayer.erase(neuron.first);
			}
		}


		/* // Clear any existing layers
		network.layers.clear();

		// Initialize a queue for BFS
		std::queue<Neuron*> neuronQueue;

		std::unordered_map<Neuron*, Neuron::ID> Neuron::IDs;
		for (auto& neuron : network.neurons)
			Neuron::IDs[neuron.second] = neuron.first;

		// Mark all neurons as unvisited
		std::unordered_map<Neuron::ID, bool> visited;
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
		std::unordered_map<Neuron::ID, size_t> neuronLayerMap;

		// Perform BFS to split neurons into layers
		size_t currentLayerIndex = 0;
		while (!neuronQueue.empty()) {
			Layer layer;
			size_t currentLayerSize = neuronQueue.size();
			for (size_t i = 0; i < currentLayerSize; ++i) {
				Neuron* currentNeuron = neuronQueue.front();
				neuronQueue.pop();
				layer.neurons.push_back(currentNeuron);
				neuronLayerMap[Neuron::IDs[currentNeuron]] = currentLayerIndex;

				// Add input connections to the current layer
				for (Connection* connection : network.connections) {
					if (connection->getEndNeuron() == currentNeuron) {
						if (neuronLayerMap[Neuron::IDs[connection->getStartNeuron()]] < currentLayerIndex) {
							// Input neuron belongs to a previous layer, add connection to current layer
							layer.inputConnections.push_back(connection);
						}
					}
					if (connection->getStartNeuron() == currentNeuron) {
						Neuron* inputNeuron = connection->getEndNeuron();
						// Add input neuron to queue if not visited yet
						if (!visited[Neuron::IDs[inputNeuron]]) {
							neuronQueue.push(inputNeuron);
							visited[Neuron::IDs[inputNeuron]] = true;
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

		std::unordered_map<Neuron*, Neuron::ID> Neuron::IDs;
		for (auto& neuron : network.neurons)
			Neuron::IDs[neuron.second] = neuron.first;

		// Mark all neurons as unvisited
		std::unordered_map<Neuron::ID, bool> visited;
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
						if (!visited[Neuron::IDs[inputNeuron]])
						{
							neuronQueue.push(inputNeuron);
							visited[Neuron::IDs[inputNeuron]] = true;
						}
					}
				}
			}
			network.layers.push_back(layer);
		}*/
	}

	void CustomConnectedNeuralNet::CustomConnectedNeuralNetBuilder::removeDuplicateConnections(
		const std::vector<ConnectionInfo>& connectionsIn,
		std::vector<ConnectionInfo>& connectionsOut)
	{
		std::unordered_map<std::string, ConnectionInfo> uniqueConnections;
		for (const auto& connection : connectionsIn)
		{
			std::string key = std::to_string(connection.fromNeuronID) + "-" + std::to_string(connection.toNeuronID);
			if (uniqueConnections.find(key) == uniqueConnections.end())
				uniqueConnections[key] = connection;
		}

		connectionsOut.clear();
		connectionsOut.reserve(uniqueConnections.size());
		for (const auto& entry : uniqueConnections)
			connectionsOut.push_back(entry.second);
	}

	void CustomConnectedNeuralNet::CustomConnectedNeuralNetBuilder::sortLayers(NetworkData& network)
	{
		for (auto& layer : network.layers)
		{
			std::sort(layer.neurons.begin(), layer.neurons.end(), [](Neuron* a, Neuron* b) {
				return a->getID() < b->getID();
			});
		}
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
			unsigned int currentNeuron::ID = q.front();
			q.pop();

			// Add neuron to the current layer
			layers.back().push_back({ 0, currentNeuron::ID, 0.0f });

			// Visit all neighbors of current neuron
			for (unsigned int neighbor : adjacencyList.at(currentNeuron::ID)) {
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