#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Função de ativação sigmóide
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivada da função sigmóide
double sigmoidDerivative(double x) {
    return x * (1 - x);
}

struct Neuron {
    double value;
    vector<double> weights;
};

class PerceptronNetwork {
private:
    vector<vector<Neuron>> layers;

public:
    PerceptronNetwork(int inputSize, int hiddenSize, int outputSize) {
        srand(static_cast<unsigned>(time(nullptr)));
        
        layers.push_back(vector<Neuron>(inputSize + 1)); // +1 para o bias

        // Inicializa pesos da camada de entrada para a oculta
        for (auto &neuron : layers[0]) {
            neuron.weights.resize(hiddenSize);
            for (double &weight : neuron.weights) {
                weight = ((double) rand() / (RAND_MAX)) * 2 - 1; // Peso aleatório entre -1 e 1
            }
        }

        // Cria camada oculta
        layers.push_back(vector<Neuron>(hiddenSize + 1)); // +1 para o bias

        // Inicializa pesos da camada oculta para a de saída
        for (auto &neuron : layers[1]) {
            neuron.weights.resize(outputSize);
            for (double &weight : neuron.weights) {
                weight = ((double) rand() / (RAND_MAX)) * 2 - 1; // Peso aleatório entre -1 e 1
            }
        }

        // Cria camada de saída
        layers.push_back(vector<Neuron>(outputSize));
    }

    // Função para alimentar a rede com entradas e obter saídas
    vector<double> feedForward(const vector<double> &inputs) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            layers[0][i].value = inputs[i];
        }
        // Bias para a camada de entrada
        layers[0].back().value = 1.0;

        for (size_t layerIdx = 1; layerIdx < layers.size(); ++layerIdx) {
            for (size_t neuronIdx = 0; neuronIdx < layers[layerIdx].size(); ++neuronIdx) {
                double sum = 0.0;
                for (size_t prevNeuronIdx = 0; prevNeuronIdx < layers[layerIdx - 1].size(); ++prevNeuronIdx) {
                    sum += layers[layerIdx - 1][prevNeuronIdx].value * 
                           layers[layerIdx - 1][prevNeuronIdx].weights[neuronIdx];
                }
                layers[layerIdx][neuronIdx].value = sigmoid(sum);
            }
            // Bias para a camada oculta
            if (layerIdx < layers.size() - 1) {
                layers[layerIdx].back().value = 1.0;
            }
        }

        vector<double> outputs;
        for (auto &neuron : layers.back()) {
            outputs.push_back(neuron.value);
        }

        return outputs;
    }
};

int main() {
    PerceptronNetwork network(2, 3, 1);

    vector<double> inputs = {0.5, 0.8};

    vector<double> outputs = network.feedForward(inputs);

    cout << "Output: " << outputs[0] << endl;

    return 0;
}
